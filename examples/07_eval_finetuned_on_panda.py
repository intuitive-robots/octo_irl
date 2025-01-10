"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
from envs.real_robot_env import convert_obs, RREnvClient
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

# Set app flags
FLAGS = flags.FLAGS

# Model flags
flags.DEFINE_string("checkpoint_weights_path", None, "Path to checkpoint", required=True)
# TODO Change unnormalization, use https://huggingface.co/rail-berkeley/octo-small-1.5/blob/main/dataset_statistics.json as reference
flags.DEFINE_string("unnorm_key", "bridge_dataset", "Key of the dataset in dataset_statistics.json")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step")
flags.DEFINE_bool("deterministic", False, "Model parameter")
flags.DEFINE_float("temperature", 1.0, "Model parameter")

# Custom to real robot env clinet
flags.DEFINE_string("ip", "127.0.0.1", "IP address of the real robot env server")
flags.DEFINE_integer("port", 6060, "Port of the real robot env server")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
# Currently, the start position can only be defined when starting the real robot env server
#flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
# Currently not supported, should be non-blocking per default
#flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("action_horizon", 4, "Length of action sequence to execute/ensemble")
flags.DEFINE_integer("sticky_gripper_num_steps", 1, "Number of steps for sticky gripper logic to grasp")

# Show image flag
flags.DEFINE_bool("show_image", False, "Show image")

# --------------------------------------------------------------------------------
# Original note about STEP_DURATION:
# Bridge data was collected with non-blocking control and a step duration of 0.2s.
# However, we relabel the actions to make it look like the data was collected with
# blocking control and we evaluate with blocking control.
# Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
# --------------------------------------------------------------------------------
# At the moment the Panda evaluation uses non blocking commands, therefore this should probably not be touched.
STEP_DURATION = 0.2


def main(_):

    # Set up the real robot env client
    env_client = RREnvClient(
        name="Octo Client",
        host=FLAGS.ip,
        port=FLAGS.port,
        sticky_gripper_num_steps= FLAGS.sticky_gripper_num_steps,
    )
    env_client.connect()

    # Load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # Wrap the robot environment
    env = HistoryWrapper(env_client, FLAGS.window_size)
    env = TemporalEnsembleWrapper(env, FLAGS.action_horizon)
    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    # env = RHCWrapper(env, FLAGS.action_horizon)

    # Create policy functions
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        argmax,
        temperature,
        rng,
    ):
        # Add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            unnormalization_statistics=pretrained_model.dataset_statistics[
                FLAGS.unnorm_key
            ]["action"],
            argmax=argmax,
            temperature=temperature,
        )
        # Remove batch dim
        return actions[0]

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            argmax=FLAGS.deterministic,
            temperature=FLAGS.temperature,
        )
    )

    im_shape = env_client.observation_space["image_primary"].shape
    goal_image = jnp.zeros(im_shape, dtype=np.uint8)
    goal_instruction = ""

    # Goal sampling loop
    while True:
        modality = click.prompt(
            "Language, goal image or quit?", type=click.Choice(["l", "g", "q"])
        )

        if modality == "q":

            break

        elif modality == "g":

            # TODO Not needed at the moment, otherwise either fix code below or input image path instead
            raise NotImplementedError()

            if click.confirm("Take a new goal?", default=True):
                assert isinstance(FLAGS.goal_eep, list)
                assert len(FLAGS.goal_eep) == 6

                # Goal EEF position with open gripper
                goal_eep = np.array(FLAGS.goal_eep + [1], dtype=np.float64)

                # TODO Move to goal_eep
                # move_status = None
                # while move_status != WidowXStatus.SUCCESS:
                #     move_status = widowx_client.move(goal_eep, duration=1.5)

                input("Press [Enter] when ready for taking the goal image. ")
                # TODO Get observation
                obs = convert_obs(obs, FLAGS.im_size)
                goal = jax.tree_map(lambda x: x[None], obs)

            # Format task for the model
            task = model.create_tasks(goals=goal)
            # For logging purposes
            goal_image = goal["image_primary"][0]
            goal_instruction = ""

        elif modality == "l":

            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")
            # Format task for the model
            task = model.create_tasks(texts=[text])
            # For logging purposes
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)

        else:

            raise NotImplementedError()

        input("Press [Enter] to start.")

        # Reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        # Rollout
        last_tstep = time.time()
        images = [obs["image_primary"][-1]]
        goals = [goal_image]
        trajectory = [obs["proprio"]]
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # Get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("Forward pass time: ", time.time() - forward_pass_time)

                # Perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("Step time: ", time.time() - start_time)

                # Save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                # Save joint positions
                trajectory.append(obs["proprio"])

                t += 1

                if truncated:
                    break

        # TODO save trajectory

        # Save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)

    # Close env
    env_client.close()

if __name__ == "__main__":
    app.run(main)
