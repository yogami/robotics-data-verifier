import time
import sys
import os
import signal
import re

def tail_and_watch(log_file, target_pid):
    print(f"Watchdog: Starting watchdog for PID {target_pid} monitoring {log_file}...")
    
    # Check if PID exists
    try:
        os.kill(target_pid, 0)
    except OSError:
        print(f"Watchdog: PID {target_pid} does not exist. Exiting.")
        sys.exit(1)
        
    try:
        with open(log_file, "r") as f:
            # Go to the end of the file
            f.seek(0, os.SEEK_END)
            
            nan_count = 0
            while True:
                # Check if process is still alive
                try:
                    os.kill(target_pid, 0)
                except OSError:
                    print("Watchdog: Target process finished. Exiting.")
                    break
                    
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                    
                # Search for loss in LeRobot style outputs
                if "loss" in line.lower():
                    # Match nan, inf (with optional sign) or floating point/scientific notation
                    match = re.search(r"loss[:=\s]+([+-]?(?:nan|inf)|[+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)", line.lower())
                    if match:
                        val_str = match.group(1)
                        is_diverged = False
                        
                        if "nan" in val_str or "inf" in val_str:
                            is_diverged = True
                            print(f"Watchdog: Warning! Detected abnormal loss token '{val_str}' ({nan_count + 1}/3).")
                        else:
                            try:
                                val = float(val_str)
                                if val > 10.0:
                                    is_diverged = True
                                    print(f"Watchdog: Warning! Exploding loss detected: {val:.4f} > 10.0 ({nan_count + 1}/3).")
                            except ValueError:
                                pass
                                
                        if is_diverged:
                            nan_count += 1
                            if nan_count >= 3:
                                print(f"Watchdog: KILLED process {target_pid} due to consecutive exploding/NaN/Inf losses. Budget protected.")
                                os.kill(target_pid, signal.SIGKILL)
                                sys.exit(1)
                        else:
                            nan_count = 0  # Reset on valid normal loss
                        
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Watchdog: Interrupted. Exiting.")
    except Exception as e:
        print(f"Watchdog: Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python watchdog.py <log_file> <target_pid>")
        sys.exit(1)
    tail_and_watch(sys.argv[1], int(sys.argv[2]))
