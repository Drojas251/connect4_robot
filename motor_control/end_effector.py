import time
import threading
import serial


class EndEffectorClient:
    """
    Python controller for Arduino EndEffector serial interface.

    Arduino commands:
        DISPENSE
        RELOAD
        STATUS

    Arduino responses:
        STATE READY
        STATE DISPENSING
        STATE RELOADING
        STATE ERROR
    """

    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = None
        self.running = False
        self.reader_thread = None

        self.state = "UNKNOWN"
        self.last_line = ""

        self.lock = threading.Lock()

    # ---------------------------------
    # Connection
    # ---------------------------------
    def connect(self):
        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout
        )

        # Arduino reset delay
        time.sleep(2.0)

        self.running = True
        self.reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )
        self.reader_thread.start()

    def disconnect(self):
        self.running = False

        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()

    # ---------------------------------
    # Reader thread
    # ---------------------------------
    def _reader_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                with self.lock:
                    self.last_line = line

                print("[ARDUINO]", line)

                if line.startswith("STATE "):
                    new_state = line.replace("STATE ", "").strip()

                    with self.lock:
                        self.state = new_state

            except Exception as e:
                print("Reader thread error:", e)
                break

    # ---------------------------------
    # Write command
    # ---------------------------------
    def send_command(self, cmd: str):
        msg = cmd.strip() + "\n"
        self.ser.write(msg.encode())

    # ---------------------------------
    # Public API
    # ---------------------------------
    def dispense(self):
        self.send_command("DISPENSE")

    def reload(self):
        self.send_command("RELOAD")

    def request_status(self):
        self.send_command("STATUS")

    def get_state(self):
        with self.lock:
            return self.state
        
    def wait_for_state(self, target_state, timeout=5.0):
        start = time.time()

        while time.time() - start < timeout:
            if self.get_state() == target_state:
                return True
            time.sleep(0.02)

        return False

    def wait_until_ready(self, timeout=5.0):
        return self.wait_for_state("READY", timeout)

    def dispense_and_wait(self, timeout=5.0):
        self.dispense()
        status = self.wait_for_state("DISPENSING", timeout)
        if not status:
            print("Failed to start dispensing within timeout")
            return False
        return self.wait_until_ready(timeout)

    def reload_and_wait(self, timeout=5.0):
        self.reload()
        status = self.wait_for_state("RELOADING", timeout)
        if not status:
            print("Failed to start reloading within timeout")
            return False
        return self.wait_until_ready(timeout)


# ====================================================
# Example usage
# ====================================================

if __name__ == "__main__":
    ee = EndEffectorClient("/dev/ttyACM0", 115200)

    try:
        ee.connect()

        time.sleep(1.0)
        ee.request_status()

        print("Current State:", ee.get_state())

        print("\nReloading...")
        ee.reload_and_wait()

        #time.sleep(1.5)

        print("\nDispensing chip...")
        ee.dispense_and_wait()

        print("State:", ee.get_state())

        #time.sleep(1.5)

        print("\nReloading...")
        ee.reload_and_wait()

        print("State:", ee.get_state())

        #time.sleep(1.5)

    finally:
        ee.disconnect()