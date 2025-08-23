from monitor import *



if __name__ == "__main__":
    try:
        print("=== monitor memory start ===")
        cpu_monitor_thread, cpu_stop_event = start_cpu_monitoring()
        gpu_monitor_thread, gpu_stop_event = start_gpu_monitoring()
        while True:
            continue
    except KeyboardInterrupt:
        stop_cpu_monitoring(cpu_monitor_thread, cpu_stop_event)
        stop_gpu_monitoring(gpu_monitor_thread, gpu_stop_event)
        print("=== release processing ===")

