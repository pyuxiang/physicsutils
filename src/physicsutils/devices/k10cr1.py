#!/usr/bin/env python3
# Justin, 2022-07-19
#
# Communicates with Thorlabs K10CR1/M stepper motor directly,
# without use of Windows-dependent 'APT.dll' required in 'thorlabs-apt' library.
# This allows Linux platforms to interact with the motor.
#
# Uses 'thorlabs_apt_protocol' for generating and reading messages.
# Documentation for APT protocol can be found in:
# https://www.thorlabs.com/drawings/c2590feb3514326e-D78FE62A-B46C-35CB-80E66667DC083047/K10CR1_M-ManualforAPT.pdf

import serial
import warnings

import thorlabs_apt_protocol as _apt

class ThorlabsK10CR1:
    """Wrapper for APT commands relevant to K10CR1."""

    def __init__(self, port, blocking=True):
        """Creates the motor instance.

        If blocking is set to 'True', both homing and moving
        commands will block until completion. Spinning is
        non-blocking by default.

        Important: Ensure that the homing parameter 'offset_position'
        is correct for your stage.

        For usage, see main script at the bottom of the file.
        """
        self._com = serial.Serial(
            port=port,
            baudrate=115200,
            rtscts=True,
            timeout=0.1,
        )

        # Enable RTS control flow (technically not needed)
        # and reset input/output buffers
        self._com.rts = True
        self._com.reset_input_buffer()
        self._com.reset_output_buffer()
        self._com.rts = False

        # Reader for device output - unpacks messages
        self.reader = apt.Unpacker(self._com)

        # Device constants
        self.steps_per_revolution = 49_152_000
        self.blocking = blocking

        # Required initialization as per APT protocol
        self.w(apt.hw_no_flash_programming())

        # Initialize device defaults, defined below in 'Parameters'
        self.params_velocity = {}
        self.params_jog = {}
        self.params_homing = {}  # important: check homing offset
        self.params_limswitch = {}



    ################
    #   COMMANDS   #
    ################

    # Most usage will only require interfacing via these commands.
    #   For customizing operating parameters, see 'Parameters' section.
    #   For direct interfacing with motor, see 'Helper' section.

    @property
    def identity(self) -> int:
        """Returns serial number of motor.

        Note that Thorlabs software typically distinguish devices
        by serial number.
        """
        return type(self.rw(apt.hw_req_info()).serial_number)

    def home(self):
        """Requests motor to go to home based on homing parameters."""
        self.w(apt.mot_move_home())
        if self.blocking:
            while self.is_moving(): pass

    def stop(self, abrupt=False):
        """Stops motor movement."""
        self.w(apt.mot_move_stop(0x01 if abrupt else 0x02))

    def spin(self, clockwise=True):
        """Continuously moves motor (non-blocking) based on velocity parameters."""
        self.w(apt.mot_move_velocity(0x02 if clockwise else 0x01))

    @property
    def angle(self) -> float:
        """Returns angular position in degrees."""
        return self.angle_absolute % 360

    @angle.setter
    def angle(self, value: float):
        self.move(self.posdiff(self.deg2pos(value), self.position))
        if self.blocking:
            while self.is_moving(): pass

    def is_moving(self) -> bool:
        """Queries if motor is still rotating."""
        return len(set([self.position for _ in range(2)])) != 1



    ########################
    #   HELPER FUNCTIONS   #
    ########################

    # Note: Forward corresponds to counter-clockwise rotation, so
    #       reverse corresponds to clockwise rotation.
    #
    # There are 49,152,000 microsteps per revolution.
    #
    # Initial settings via Kinesis sets:
    # - Homing: Reverse direction with hardware reverse limit switch
    #           Zero offset = 4.0deg, velocity = 10.0deg/s
    # - Hardware limit switch: CCW makes (home), CW ignore.
    # - Software limit switch policy: Complete moves only
    # - Motor: Steps/revolution 200, gearbox ratio 120:1
    # - Backlash: 1deg

    def w(self, command: bytes):
        """Writes byte command to motor."""
        self._com.write(command)

    def r(self):
        """Reads from motor.

        As far as possible, we want to keep only a single message in buffer,
        for ease of debugging, and since use cases for multiple REQs before
        a single GET is rare, e.g. high resolution motor positioning.

        If multiple messages exist, a warning is raised and only the latest
        message is returned.

        Note that the status update messages pushed by the motor is generally
        ignored, since similar functionality is achieved by actually
        querying the motor directly.
        """
        resp = [msg for msg in self.reader]

        # Filter status update responses
        status = ["mot_move_completed", "mot_move_stopped", "mot_move_homed"]
        responses = [m for m in resp if m.msg not in status]
        if not responses:
            raise ValueError("No reply from motor.")

        if len(responses) > 1:
            warning = [
                "Multiple messages received:",
                *responses,
            ]
            warnings.warn("\n  * ".join(warning))

        return responses[-1]

    def rw(self, command: bytes) -> list:
        """Writes and reads motor controller."""
        self.w(command)
        return self.r()

    def pos2deg(self, position: int) -> float:
        return position / self.steps_per_revolution * 360

    def deg2pos(self, degree: float) -> int:
        return round(degree / 360 * self.steps_per_revolution)

    def posdiff(self, target_position, curr_position):
        """Returns relative move distance based on absolute position."""

        # Normalize
        target = target_position % self.steps_per_revolution
        curr = curr_position % self.steps_per_revolution

        # Get smallest rotation
        diff = target - curr
        if abs(diff) > self.steps_per_revolution/2:
            if diff > 0:
                return diff - self.steps_per_revolution
            else:
                return diff + self.steps_per_revolution
        return diff

    @property
    def angle_absolute(self):
        """Returns absolute angular position since homing.

        Generally not used since only the actual angle is
        relevant. Might be useful when excessive number of
        revolutions are performed -> warn user.
        """
        return self.position * 360 / self.steps_per_revolution

    @angle_absolute.setter
    def angle_absolute(self, value: float):
        self.goto(self.deg2pos(value))

    @property
    def position(self):
        """Returns absolute position based on encoder value.

        Accurate only after homing. Does not normalize within
        revolution steps.

        Equivalent representation using encoder_count:
        `self.rw(apt.mot_req_enccounter()).encoder_count`
        """
        return self.rw(apt.mot_req_poscounter()).position

    def move(self, distance: int = 0):
        """Move motor along 'distance' in units of microsteps."""
        self.w(apt.mot_move_relative(distance))

    def goto(self, position: int = 0):
        """Move motor to 'position' in units of microsteps."""
        self.w(apt.mot_move_absolute(position))

    def jog(self, direction: int):
        """Performs a jog based on jog parameters.

        Currently not used, similar functionality to spin.

        Args:
            direction: 0x01 (forward) or 0x02 (backward)
        """
        self.w(apt.mot_move_jog(direction))

    def identify(self):
        """Blinks LED."""
        self.w(apt.mod_identify())



    ##################
    #   PARAMETERS   #
    ##################

    # Units are in microsteps, see 'position()' for details.

    @staticmethod
    def extract(message, *fields):
        return { key:getattr(message,key) for key in fields}

    @property
    def backlash(self):
        """Returns backlash correction distance.

        Typically 1 deg == 136,533 microsteps.

        Usage:
            >>> motor.backlash = 136000
            >>> motor.backlash
            136000
        """
        return self.rw(apt.mot_req_genmoveparams()).backlash_distance

    @backlash.setter
    def backlash(self, distance: int = 136533):
        self.w(apt.mot_set_genmoveparams(distance))

    @property
    def params_velocity(self):
        """Returns trapezoidal velocity parameters used for move.

        These parameters are used during general move, move_velocity,
        and homing (prior to hitting limit switch). Empirical
        measurements find that 73301283 corresponds to a
        10 deg/s rotation.

                       ^  Factory ^  Kinesis ^
        | min_velocity |        0 |        0 |
        | acceleration |    15020 |    15011 |
        | max_velocity | 73300775 | 73286848 |

        Usage:
            >>> motor.params_velocity = {"acceleration": 15000}
            >>> motor.params_velocity
            {
                "min_velocity": 0,
                "acceleration": 15000,     # newly set
                "max_velocity": 73301283,  # see setter defaults
            }
        """
        data = self.rw(apt.mot_req_velparams())
        return self.extract(
            data, "min_velocity", "acceleration", "max_velocity",
        )

    @params_velocity.setter
    def params_velocity(self, value):
        settings = {
            "min_velocity": 0,
            "acceleration": 15011,
            "max_velocity": 73301283,
            **value,  # override by value
        }
        return self.w(apt.mot_set_velparams(**settings))

    @property
    def params_jog(self):
        """Returns jog parameters.

          - 'jog_mode': 1 (continuous), 2 (single step)
          - 'stop_mode': 1 (immediate), 2 (controlled deceleration)

                       ^  Factory ^   Kinesis ^
        | jog_mode     |     0x02 |      0x02 |
        | step_size    |   136533 |    682667 |
        | min_velocity |        0 |         0 |
        | acceleration |     7506 |     22530 |
        | max_velocity | 36643424 | 109951163 |
        | stop_mode    |     0x02 |      0x02 |
        """
        data = self.rw(apt.mot_req_jogparams())
        return self.extract(
            data,
            "jog_mode", "step_size",
            "min_velocity", "acceleration", "max_velocity",
            "stop_mode",
        )

    @params_jog.setter
    def params_jog(self, value):
        settings = {
            "jog_mode": 0x02,
            "step_size": 682667,  # 5 degrees
            "min_velocity": 0,
            "acceleration": 22530,
            "max_velocity": 109951163,
            "stop_mode": 0x02,
            **value,  # override by value
        }
        return self.w(apt.mot_set_jogparams(**settings))

    @property
    def params_homing(self):
        """Returns homing parameters, typically stage-specific.

        Avoid setting a large offset_distance, since a slower
        rate of revolution is used to accurately move to the
        home position.

                          ^ Factory ^ Kinesis ^
        | home_dir        |    0x02 |    0x02 |
        | limit_switch    |    0x01 |       - |  # undocumented
        | home_velocity   |   73290 |       - |
        | offset_distance |  819200 |       - |
        """
        data = self.rw(apt.mot_req_homeparams())
        return self.extract(
            data,
            "home_dir", "limit_switch",
            "home_velocity", "offset_distance",
        )

    @params_homing.setter
    def params_homing(self, value):
        settings = {
            "home_dir": 0x02,
            "limit_switch": 0x01,
            "home_velocity": 73301283,  # align with velocity param
            "offset_distance": 819200,
            **value,  # override by value
        }
        return self.w(apt.mot_set_homeparams(**settings))

    @property
    def params_limswitch(self):
        """Returns limit switch parameters.

        Unknown whether software limit switch is used.

        For the hardware limit switches:
          - '0x01': ignore switch
          - '0x02': switch makes on contact
          - '0x03': switch breaks on contact
          - '0x04': switch makes on contact (only used for homing)
          - '0x05': switch breaks on contact (only used for homing)
        """
        data = self.rw(apt.mot_req_limswitchparams())
        return self.extract(
            data,
            "cw_hardlimit", "ccw_hardlimit",
            "cw_softlimit", "ccw_softlimit", "soft_limit_mode",
        )

    @params_limswitch.setter
    def params_limswitch(self, value):
        settings = {
            "cw_hardlimit": 2,
            "ccw_hardlimit": 2,
            "cw_softlimit": 0,
            "ccw_softlimit": 0,
            "soft_limit_mode": 1,
            **value,  # override by value
        }
        return self.w(apt.mot_set_limswitchparams(**settings))

class Apt:
    """
    See Thorlabs APT protocol specification, page 12, for description of dest and source.
    Used when messages are broadcasted to sub-modules.
    In modern systems, typically connect one-to-one COM port connection,
    so can simply replace all source reference to 0x01 (host) and
    destination reference to 0x50 (generic USB).

    Source:
        https://www.thorlabs.com/software/apt/APT_Communications_Protocol_Rev_15.pdf
    """

    def __getattr__(self, name):
        """Intercepts calls to and injects apt.functions.

        Calls to every other function in apt is transparently passed
        through, unless they are defined in apt.functions.

        This monkey patch curries '0x50' as destination byte and
        '0x01' as source byte, since all apt.functions.[^_]* have them
        as first and second positional arguments.
        """
        target = getattr(_apt, name)  # could have been imported downstream
        if not hasattr(_apt.functions, name):
            return target

        # Patch APT function
        curry = [0x50, 0x01]
        if name.startswith("mot_") or name.startswith("mod_"):
            curry.append(0x01)  # always channel 1

        def f(*args, **kwargs):
            target = getattr(_apt, name)
            return target(*curry, *args, **kwargs)
        return f

apt = Apt()  # required

# Quick tests
if __name__ == "__main__":
    import time
    port = "/dev/ttyUSB0"  # changeme

    # Connect to motor
    motor = ThorlabsK10CR1(port)
    motor.identify()                  # blinks LED
    print("Serial:", motor.identity)  # prints identity information

    # Perform rotations
    motor.stop()              # stop any existing rotations
    motor.home()              # go to home, non-blocking
    while motor.is_moving():  # wait until homing completed
        pass

    # Perform blocking rotations
    motor.blocking = True
    motor.home()              # go to home, blocking
    motor.angle = 45          # go to 45 degree position
    motor.angle += 45         # go to 90 degree position

    # Perform continuous jog
    motor.spin(clockwise=True)  # turn clockwise perpetually, non-blocking
    time.sleep(3)
    print(motor.angle)          # motor angle after 3 seconds
    motor.stop()
