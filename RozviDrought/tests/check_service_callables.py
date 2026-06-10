import sys
from pathlib import Path
import inspect

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

MODULES = [
    "app.services.feature_service",
    "app.services.subsystem_service",
    "app.services.fusion_service",
    "app.services.inference_service",
]

def main():
    for module_name in MODULES:
        print("=" * 80)
        print("MODULE:", module_name)
        mod = __import__(module_name, fromlist=["*"])

        names = [n for n in dir(mod) if not n.startswith("_")]
        print("\nPUBLIC NAMES")
        for name in names:
            obj = getattr(mod, name)
            kind = "other"
            if inspect.isfunction(obj):
                kind = "function"
            elif inspect.isclass(obj):
                kind = "class"
            print(f"- {name} [{kind}]")

        print("\nCALLABLE SIGNATURES")
        for name in names:
            obj = getattr(mod, name)
            if inspect.isfunction(obj):
                try:
                    print(f"def {name}{inspect.signature(obj)}")
                except Exception:
                    print(f"def {name}(...)")
            elif inspect.isclass(obj):
                print(f"class {name}")
                for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if meth_name.startswith("_"):
                        continue
                    try:
                        print(f"  - {meth_name}{inspect.signature(meth)}")
                    except Exception:
                        print(f"  - {meth_name}(...)")
        print()

if __name__ == "__main__":
    main()