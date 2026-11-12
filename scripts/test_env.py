"""
Environment Verification and Diagnostic Script.
Inspects Python, CPU, RAM, GPU (hardware and PyTorch CUDA), Hugging Face libraries,
and executes a small functional sanity test without downloading large model weights.
Exports diagnostic report to results/environment_info.json.
"""

import sys
import os
import platform
import subprocess
import json
import time

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def get_system_ram():
    """Query system RAM using ctypes on Windows."""
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_gb = stat.ullTotalPhys / (1024 ** 3)
        avail_gb = stat.ullAvailPhys / (1024 ** 3)
        used_gb = total_gb - avail_gb
        return {
            "total_gb": round(total_gb, 2),
            "available_gb": round(avail_gb, 2),
            "used_gb": round(used_gb, 2),
            "load_percent": stat.dwMemoryLoad
        }
    except Exception as e:
        return {"error": str(e)}


def get_nvidia_gpu_info():
    """Query NVIDIA GPU hardware details via nvidia-smi."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.free,driver_version',
            '--format=csv,noheader,nounits'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0],
                    "total_vram_mib": float(parts[1]),
                    "free_vram_mib": float(parts[2]),
                    "driver_version": parts[3]
                })
        return {"detected": True, "count": len(gpus), "gpus": gpus}
    except FileNotFoundError:
        return {"detected": False, "reason": "nvidia-smi binary not found in PATH"}
    except Exception as e:
        return {"detected": False, "reason": str(e)}


def check_pytorch():
    """Inspect PyTorch installation and CUDA availability."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_avail,
            "cuda_version_torch": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            "device_count": torch.cuda.device_count() if cuda_avail else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else None
        }
        return info
    except ImportError:
        return {"installed": False, "cuda_available": False}


def check_library_versions():
    """Check availability and versions of critical project dependencies."""
    libs = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "transformers",
        "accelerate",
        "datasets",
        "evaluate",
        "sacrebleu",
        "sentencepiece"
    ]
    results = {}
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "unknown")
            results[lib] = {"available": True, "version": ver}
        except ImportError:
            results[lib] = {"available": False, "version": None}
    return results


def run_functional_sanity_test():
    """
    Execute a minimal functional sanity test:
    1. Tensor creation and operations (CPU / CUDA if available).
    2. SciKit-Learn Macro-F1 computation.
    3. SacreBLEU / chrF++ metric execution.
    4. Data loader sanity check on 5 SciFact claims.
    """
    test_results = {}
    
    # Test 1: Tensor operations
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.ones((10, 10), device=device)
        y = x @ x
        test_results["tensor_ops"] = {
            "status": "PASS",
            "device_used": device,
            "output_shape": list(y.shape),
            "sample_value": float(y[0, 0].item())
        }
    except Exception as e:
        test_results["tensor_ops"] = {"status": "FAIL", "error": str(e)}
        
    # Test 2: Scikit-learn Macro-F1
    try:
        from sklearn.metrics import f1_score
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 2]
        f1 = f1_score(y_true, y_pred, average="macro")
        test_results["sklearn_metrics"] = {
            "status": "PASS",
            "calculated_macro_f1": round(float(f1), 4)
        }
    except Exception as e:
        test_results["sklearn_metrics"] = {"status": "FAIL", "error": str(e)}

    # Test 3: SacreBLEU / chrF
    try:
        import sacrebleu
        ref = "The clinical trial confirmed drug efficacy."
        hyp = "The clinical study demonstrated drug efficacy."
        chrf = sacrebleu.sentence_chrf(hyp, [ref])
        test_results["sacrebleu_metric"] = {
            "status": "PASS",
            "sample_chrf_score": round(chrf.score, 2)
        }
    except Exception as e:
        test_results["sacrebleu_metric"] = {"status": "FAIL", "error": str(e)}

    # Test 4: Data loader sanity on SciFact dev split
    try:
        from src.data_loader import load_jsonl, parse_claim_evidence
        dev_path = os.path.join(PROJECT_ROOT, "data", "claims_dev.jsonl")
        claims = load_jsonl(dev_path)
        sample_5 = claims[:5]
        parsed = [parse_claim_evidence(c) for c in sample_5]
        test_results["scifact_data_loader"] = {
            "status": "PASS",
            "loaded_claims_count": len(claims),
            "sample_parsed_evidence_count": sum(len(p) for p in parsed)
        }
    except Exception as e:
        test_results["scifact_data_loader"] = {"status": "FAIL", "error": str(e)}

    return test_results


def main():
    print("=" * 70)
    print("      SCIENTIFIC FACTUALITY PROJECT: ENVIRONMENT SANITY CHECK")
    print("=" * 70)
    
    # 1. Platform & CPU
    print("\n[1] Platform & Host Hardware:")
    print(f"    • Operating System : {platform.platform()}")
    print(f"    • Python Version   : {sys.version.split()[0]}")
    print(f"    • Architecture     : {platform.machine()}")
    print(f"    • Processor        : {platform.processor()}")
    
    # 2. System RAM
    ram = get_system_ram()
    print("\n[2] Physical System Memory (RAM):")
    if "error" not in ram:
        print(f"    • Total RAM        : {ram['total_gb']} GB")
        print(f"    • Available RAM    : {ram['available_gb']} GB")
        print(f"    • Memory Load      : {ram['load_percent']}%")
    else:
        print(f"    • RAM Query Error  : {ram['error']}")
        
    # 3. GPU Hardware
    gpu_hw = get_nvidia_gpu_info()
    print("\n[3] NVIDIA GPU Hardware (via nvidia-smi):")
    if gpu_hw["detected"]:
        for i, g in enumerate(gpu_hw["gpus"]):
            print(f"    • GPU {i}            : {g['name']}")
            print(f"    • Total VRAM       : {g['total_vram_mib']} MiB (~{round(g['total_vram_mib']/1024, 2)} GB)")
            print(f"    • Free VRAM        : {g['free_vram_mib']} MiB")
            print(f"    • Driver Version   : {g['driver_version']}")
    else:
        print(f"    • GPU Status       : Not Detected ({gpu_hw.get('reason', 'N/A')})")
        
    # 4. PyTorch & CUDA
    torch_info = check_pytorch()
    print("\n[4] PyTorch & CUDA Runtime:")
    if torch_info["installed"]:
        print(f"    • PyTorch Version  : {torch_info['version']}")
        print(f"    • CUDA Available   : {torch_info['cuda_available']}")
        if torch_info["cuda_available"]:
            print(f"    • CUDA Device      : {torch_info['device_name']}")
        else:
            print("    • Note             : PyTorch is running CPU build (+cpu).")
            if gpu_hw["detected"]:
                print("      Hardware GPU is present; to enable GPU acceleration, install torch with CUDA support.")
    else:
        print("    • PyTorch Status   : Not installed")

    # 5. Dependency Libraries
    libs = check_library_versions()
    print("\n[5] Critical Library Versions:")
    for lib, info in libs.items():
        if info["available"]:
            print(f"    • {lib:<15}: {info['version']} [OK]")
        else:
            print(f"    • {lib:<15}: NOT INSTALLED [MISSING]")

    # 6. Functional Sanity Tests
    sanity = run_functional_sanity_test()
    print("\n[6] Functional Sanity Test Results:")
    all_pass = True
    for test_name, res in sanity.items():
        status = res.get("status", "FAIL")
        if status != "PASS":
            all_pass = False
        print(f"    • {test_name:<20}: [{status}] {res}")

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL VERDICT: [PASS] Environment is verified for dataset, translation, and evaluation!")
    else:
        print("OVERALL VERDICT: [ATTENTION REQUIRED] One or more sanity checks failed.")
    print("=" * 70)

    # Export structured JSON
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_file = os.path.join(results_dir, "environment_info.json")
    
    diagnostic_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "os": platform.platform(),
            "python": sys.version,
            "architecture": platform.machine(),
            "processor": platform.processor()
        },
        "ram": ram,
        "gpu_hardware": gpu_hw,
        "pytorch": torch_info,
        "libraries": libs,
        "sanity_tests": sanity,
        "verdict": "PASS" if all_pass else "FAIL"
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_report, f, indent=2)
    print(f"\nDiagnostic report saved to: {report_file}")


if __name__ == "__main__":
    main()
