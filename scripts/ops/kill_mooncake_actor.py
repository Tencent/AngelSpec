import os

import ray


def main() -> None:
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR")
    try:
        actor = ray.get_actor("mooncake_master")
    except ValueError:
        return
    ray.kill(actor, no_restart=True)


if __name__ == "__main__":
    main()
