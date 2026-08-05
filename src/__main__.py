import fire  # pyright: ignore[reportMissingTypeStubs]
from .cli import RAGCLI


def main() -> None:
    fire.Fire(RAGCLI)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    main()
