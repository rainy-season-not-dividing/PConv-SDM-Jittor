import importlib.util


def main():
    print("Jittor installed:", bool(importlib.util.find_spec("jittor")))
    if importlib.util.find_spec("jittor"):
        import jittor as jt

        print("Jittor version:", getattr(jt, "__version__", "unknown"))
        print("CUDA flag:", getattr(jt.flags, "use_cuda", "unknown"))
        print("Has CUDA:", getattr(jt, "has_cuda", "unknown"))


if __name__ == "__main__":
    main()

