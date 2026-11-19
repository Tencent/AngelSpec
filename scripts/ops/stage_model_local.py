import argparse
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def inventory(root):
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not (root / "config.json").is_file():
        raise ValueError("Source requires config.json")
    records = {}
    for path in files:
        stat = path.stat()
        records[str(path.relative_to(root))] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards.issubset(records):
            raise ValueError("Source checkpoint is incomplete")
    return records


def stage(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Source and destination must be separate directories")
    records = inventory(source)
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.parent / (destination.name + ".stage.json")
    lock = destination.parent / (destination.name + ".stage.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            if not receipt.exists():
                raise FileExistsError("Existing destination has no verified receipt")
            metadata = json.loads(receipt.read_text())
            if metadata["source"] != str(source) or metadata["identity"] != identity:
                raise ValueError("Cached model source identity changed")
            for name, info in metadata["files"].items():
                path = destination / name
                if (
                    not path.is_file()
                    or path.stat().st_size != info["size"]
                    or digest(path) != info["sha256"]
                ):
                    raise ValueError(f"Cached model verification failed: {name}")
            return {"status": "verified_existing", "path": str(destination)}
        required = sum(info["size"] for info in records.values())
        if shutil.disk_usage(destination.parent).free < required + (10 << 30):
            raise OSError("Insufficient local disk space")
        temporary = destination.parent / (destination.name + ".partial-" + uuid.uuid4().hex)
        temporary.mkdir()
        verified = {}
        for number, (name, info) in enumerate(records.items(), 1):
            src = source / name
            dst = temporary / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with src.open("rb") as reader, dst.open("wb") as writer:
                for block in iter(lambda: reader.read(16 << 20), b""):
                    writer.write(block)
                    checksum.update(block)
                writer.flush()
                os.fsync(writer.fileno())
            if dst.stat().st_size != info["size"] or digest(dst) != checksum.hexdigest():
                raise ValueError(f"Copy verification failed: {name}")
            verified[name] = {**info, "sha256": checksum.hexdigest()}
            print(json.dumps({"copied": number, "total": len(records), "file": name}), flush=True)
        if inventory(source) != records:
            raise ValueError("Source changed during staging; refusing publication")
        metadata = {
            "source": str(source),
            "identity": identity,
            "files": verified,
            "bytes": required,
        }
        pending_receipt = temporary / ".stage-receipt.json"
        pending_receipt.write_text(json.dumps(metadata, indent=2))
        temporary.rename(destination)
        (destination / pending_receipt.name).replace(receipt)
        return {"status": "staged_verified", "path": str(destination), "bytes": required}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.destination)), flush=True)


if __name__ == "__main__":
    main()
