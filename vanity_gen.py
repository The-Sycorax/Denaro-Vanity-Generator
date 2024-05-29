"""
MIT License

Copyright (c) 2024 The-Sycorax (https://github.com/The-Sycorax)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE+= OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import secrets
import time
import threading
import argparse
import logging 
import re
import sys
import traceback

from humanfriendly import format_timespan
from os import get_terminal_size
from collections import deque


class VanityGen:
    
    # Define constants for the secp256r1 (P-256) elliptic curve.
    P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF  # Prime field order
    A = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC  # Coefficient A in the curve equation
    B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B  # Coefficient B in the curve equation
    GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296 # x-coordinate of the base point G
    GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5 # y-coordinate of the base point G
    N = 0XFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551  # Order of the base point G
    

    # Base58 characters used for encoding addresses.
    b58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    

    # Heading message for display during address generation.
    HEADING_MESSAGE_LINES = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n",
        "┃ Denaro Vanity Gen v2.0 ┃ Developed By: The-Sycorax ┃\n",
        "┗━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
    ]


    # Combine the heading lines into a single string, prepended by escape codes to control console cursor.
    HEADING_MESSAGE = f"\x1B[{len(HEADING_MESSAGE_LINES)}A" + "".join(HEADING_MESSAGE_LINES)


    # Define the structure of the progress message for display during address generation.
    PROGRESS_MESSAGE_LINES = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n",
        "┃ - Difficulty: {}{}┃\n",  
        "┃                                                              ┃\n",
        "┃ - Addresses Generated: {}{}┃\n",  
        "┃ - Speed: {} Addr/s{}┃\n",  
        "┃ - Time elapsed: {}{}┃\n",  
        "┃ - Estimated time: {}{}┃\n",  
        "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n",
    ]


    # Combine the progress lines into a single string, prepended by escape codes to control console cursor.
    PROGRESS_MESSAGE = f"\x1B[{len(PROGRESS_MESSAGE_LINES)}A" + "".join(PROGRESS_MESSAGE_LINES)


    def __init__(self):
        """
        Initializes the VanityGen instance, setting up the tracking of time,
        count of addresses generated, and the precomputation of ECC points based on the base point G.
        """
        # Initialize statistic variables
        self.start_time = time.time()
        self.time_elapsed = 0
        self.count = 0
        self.attempts_per_second = 0
        self.run_progress_thread = True
        self.progress_thread = threading.Thread(target=self.update_progress, args=("",), daemon=True)
        self.silent = False
        
        # Initialize ECC points for batch processing.
        self.mG = [(0, 0)] * 2049  # List of tuples to store points on the curve.
        self.mGx = [0] * 2049  # List to store x-coordinates of the points.
        self.mGy = [0] * 2049  # List to store y-coordinates of the points.
        self.mGx[1], self.mGy[1] = self.GX, self.GY  # Set the base point G.
        self.mG[1] = (self.GX, self.GY)
        self.precompute_points()  # Precompute additional points for efficient computations.

        self.console_msg = ""
    

    def update_progress(self, vanity_prefix=""):
        """
        This method runs in a separate thread and continuously updates the progress of
        vanity address generation. It calculates the rate of attempts per second and
        displays real-time statistics about the progress, it updates every 0.1 seconds
        to limit resource usage.
    
        Args:
        vanity_prefix (str): The prefix being targeted for the vanity address generation.
        """
        
        while self.run_progress_thread:
            if not self.silent:
                self.time_elapsed = time.time() - self.start_time  # Compute the elapsed time since start.
                if self.time_elapsed > 0:
                    # Calculate the number of attempts per second.
                    self.attempts_per_second = self.count / self.time_elapsed
        
                # Fetch updated stats based on current progress.
                stats = self.vanity_generation_stats(vanity_prefix, int(self.attempts_per_second))
    
                # Format the expected attempts and addresses generated to adjust space dynamically.                
                if stats['expected_attempts'] < 2**64:
                    difficulty_str = "{:,.0f}".format(stats['expected_attempts']) # Formats integer value with commas
                else:
                    difficulty_str = "{:e}".format(stats['expected_attempts']) # Formats to scientific notation    
                
                addresses_str = "{:,}".format(self.count)
                speed_str = "{:,.0f}".format(self.attempts_per_second)
                time_elapsed_str = format_timespan(int(self.time_elapsed))
                estimated_time_str = stats['expected_time_readable']  # Assumes this is already a string.                
               
                if not estimated_time_str == "∞":
                    if "year" in estimated_time_str:
                        years = int(estimated_time_str.split(' year')[0]) # Extracts years value
                        if years < 2**38:
                            years = "{:,.0f}".format(years) # Formats integer value with commas
                        else:
                            years = "{:e}".format(years) # Formats to scientific notation    
                        estimated_time_str = years +" year"+ estimated_time_str.split(' year')[1] # Re-constructs estimated_time_str with formatted years value                    
                    if "second" in estimated_time_str:
                        # Extracts the seconds, rounds them, and then reconstructs the string without a decimal in the seconds
                        seconds = estimated_time_str.split(' second')
                        estimated_time_str = ' '.join(seconds[0].split()[:-1]) + f" {int(float(seconds[0].split()[-1]))}" + ' second' + seconds[1]
    
                # Calculate padding for each line to align the ending char ┃.
                padding_difficulty = " " * (64 - len(difficulty_str) - 17)
                padding_addresses = " " * (64 - len(addresses_str) - 26)
                padding_speed = " " * (64 - len(speed_str) - 19)
                padding_elapsed = " " * (64 - len(time_elapsed_str) - 19)
                padding_estimated = " " * (64 - len(estimated_time_str) - 21)
                
    
                # Assemble the progress message.
                message = "".join(self.PROGRESS_MESSAGE_LINES).format(
                    difficulty_str, padding_difficulty,
                    addresses_str, padding_addresses,
                    speed_str, padding_speed,
                    time_elapsed_str, padding_elapsed,
                    estimated_time_str, padding_estimated
                )
            
            # Assemble the heading message.
            heading_message = "".join(self.HEADING_MESSAGE_LINES)

            # Create a buffer to hold the entire update buffer
            update_buffer = []
            
            # Get terminal width
            terminal_width = get_terminal_size().columns
            
            # Adjust heading message to fit terminal width
            adjusted_heading_message = "\n".join(line[:terminal_width] for line in heading_message.split("\n"))

            # Build the update buffer
            update_buffer.append("\033c")  # Optionally clear the screen if necessary
            update_buffer.append(adjusted_heading_message + "\n")
            update_buffer.append(self.console_msg + "\n\n")
            
            if not self.silent:
                # Adjust progress message to fit terminal width
                adjusted_progress_message = "\n".join(line[:terminal_width] for line in message.split("\n"))
                update_buffer.append(adjusted_progress_message)

            # Convert buffer list to a single string
            final_output = ''.join(update_buffer)
            
            # Print the final output in one go
            print(final_output, end="\r")

            time.sleep(0.05)  # Adjust the sleep time if needed to reduce update frequency
        
    
    def vanity_generation_stats(self, pattern: str, addresses_per_second: int):
        """
        Calculates the statistics required for generating a vanity address based on the length of the desired pattern.

        Args:
        pattern (str): The vanity address pattern that the generation is aiming to match.
        addresses_per_second (int): The rate at which addresses are being generated per second.
    
        Returns:
        dict: A dictionary containing probability of match, expected number of attempts,
              estimated time in seconds, and readable estimated time for achieving the match.
    
        """
        base58_characters = 58  # The number of characters in the base58 encoding.
        first_char_probability = 1 / 2  # Probability of the first character matching (50% chance).
    
        # Calculate the probability of the rest of the pattern matching.
        if len(pattern) > 1:
            remaining_pattern_probability = (1 / base58_characters) ** (len(pattern) - 1)
        else:
            remaining_pattern_probability = 1
    
        # Overall probability of the full pattern matching.
        overall_probability = first_char_probability * remaining_pattern_probability
        expected_attempts = 1 / overall_probability  # Expected attempts required to find a match.
    
        # Guard against zero to prevent division error.
        if addresses_per_second == 0:
            addresses_per_second = 1
    
        # Calculate the expected time in seconds to achieve the match.
        expected_time_seconds = expected_attempts / addresses_per_second
    
        # Return a dictionary containing all calculated stats.
        return {
            "probability": overall_probability,
            "expected_attempts": expected_attempts,
            "expected_time_seconds": expected_time_seconds,
            "expected_time_readable": self.format_time(expected_time_seconds, 2)
        }
    

    def format_time(self, expected_time_seconds, units):
        try:
            result = format_timespan(expected_time_seconds, max_units=units)
        except:
            result = "∞"
        return result


    def precompute_points(self):
        """
        Precomputes multiples of a base point G (denoted as 2G to 2048G) on the elliptic curve. This is useful
        for speeding up subsequent cryptographic operations such as scalar multiplication. Precomputed points
        are stored in two dictionaries for x and y coordinates separately.    
        """
        # Explicit initialization of 2G with hardcoded coordinates.
        self.mG[2] = (0x7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978,
                      0x07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1)
        self.mGx[2], self.mGy[2] = self.mG[2]
        # Iteratively compute 3G to 2048G by adding G to the last computed point.
        for i in range(3, 2049):
            self.mG[i] = self.point_add(self.mG[1], self.mG[i-1])
            self.mGx[i], self.mGy[i] = self.mG[i]
    

    def inv(self, a, p):
        """
        Computes the modular multiplicative inverse of 'a' under modulo 'p' using the extended Euclidean algorithm.
    
        Args:
        a (int): The number to find the inverse of.
        p (int): The modulus.
    
        Returns:
        int: The inverse of 'a' modulo 'p'.
        """
        # Initialize variables for the extended Euclidean algorithm.
        u, v = a % p, p
        x1, x2 = 1, 0
        while u != 1:
            q, r = divmod(v, u)
            x = x2 - q * x1
            v = u
            u = r
            x2 = x1
            x1 = x
        # Ensure the result is positive.
        return x1 % p


    def inv_batch(self, x, batchx, p):
        """
        Computes the inverses of x - xi for each xi in batchx using Fermat's little theorem.
        This function is utilized when a batch of inverses is needed at once, which improves efficiency.
    
        Args:
        x (int): The base value from which differences are calculated.
        batchx (list[int]): A list of integers to compute differences and their inverses against x.
        p (int): The modulus.
    
        Returns:
        list[int]: A list of the inverses for each computed difference.
        """
        n = len(batchx)
        partial = [0] * n
        a = (batchx[0] - x) % p
        partial[0] = a
        # Compute the product of differences modulo p.
        for i in range(1, n):
            a = (a * (batchx[i] - x)) % p
            partial[i] = a
        # Compute the inverse of the last element in the partial product.
        inverse = self.inv(partial[-1], p)
        batch_inverse = [0] * n
        # Compute each inverse by using the previously computed inverses.
        for i in range(n - 1, 0, -1):
            batch_inverse[i] = (partial[i - 1] * inverse) % p
            inverse = (inverse * (batchx[i] - x)) % p
        batch_inverse[0] = inverse  # Inverse of the first element.
        return batch_inverse


    def point_add(self, p, q):
        """
        Adds two points 'p' and 'q' on an elliptic curve using the chord-and-tangent rule.
    
        Args:
        p (tuple[int, int]): The first point in the form (x1, y1).
        q (tuple[int, int]): The second point in the form (x2, y2).
    
        Returns:
        tuple[int, int]: The resulting point after addition.
        """
        # Identity element checks.
        if p == (0, 0):
            return q
        if q == (0, 0):
            return p
        # Point inversion check.
        (x1, y1), (x2, y2) = p, q
        if (x1 == x2) and (y1 != y2):
            return (0, 0)  # Points are inverses.
        # Point doubling special case.
        if p == q:
            return self.point_double(p)
        # General addition case.
        s = (y2 - y1) * self.inv(x2 - x1, self.P) % self.P
        x3 = (s**2 - x1 - x2) % self.P
        y3 = (s * (x1 - x3) - y1) % self.P
        return (x3, y3)


    def point_double(self, p):
        """
        Doubles a point on the elliptic curve. This is another core operation used,
        especially in the context of implementing scalar multiplication.
    
        Args:
        p (tuple[int, int]): The point to double.
    
        Returns:
        tuple[int, int]: The doubled point.
        """
        (x, y) = p
        if y == 0:
            return (0, 0)  # Doubling the point at infinity.
        # Use the derivative of the curve equation to find the slope.
        s = (3 * x**2 + self.A) * self.inv(2 * y, self.P) % self.P
        x3 = (s**2 - 2 * x) % self.P
        y3 = (s * (x - x3) - y) % self.P
        return (x3, y3)


    def scalar_multiply(self, k, p):
        """
        Performs scalar multiplication of a point 'p' by an integer 'k' using the double-and-add algorithm.
        This method is important for public key generation.
    
        Args:
        k (int): The scalar multiplier.
        p (tuple[int, int]): The point on the elliptic curve to be multiplied.
    
        Returns:
        tuple[int, int]: The resulting point after multiplication.
        """
        n = p
        q = (0, 0)  # Start with the identity element.
        while k:
            if k & 1:
                q = self.point_add(q, n)  # Add when the least significant bit is 1.
            n = self.point_double(n)  # Always double the point.
            k >>= 1  # Right shift k by 1.
        return q


    #Unused
    def montgomery_ladder(self, k, p):
        """
        Performs scalar multiplication using the Montgomery ladder technique, an algorithm that provides
        enhanced security against side-channel attacks. It's particularly effective in constant-time
        implementations of scalar multiplication on elliptic curves. This method is unused.
    
        Args:
        k (int): The scalar value for multiplication.
        p (tuple[int, int]): The point on the elliptic curve to be multiplied.
    
        Returns:
        tuple[int, int]: The point resulting from the scalar multiplication.
        """
        # Initialize points R0 and R1 for the Montgomery ladder steps.
        R0 = (0, 0)  # Represents the accumulated result.
        R1 = p       # Represents the point to be added.
        # Process each bit of the scalar from most significant to least significant.
        for i in reversed(range(k.bit_length())):
            if (k >> i) & 1:
                R0 = self.point_add(R0, R1)
                R1 = self.point_double(R1)
            else:
                R1 = self.point_add(R0, R1)
                R0 = self.point_double(R0)
        return R0


    def double_add_P_Q_inv(self, x1, y1, x2, y2, invx2x1):
        """
        Performs a double point addition using a precomputed inverse of the difference of x-coordinates.
        This method is used to optimize point addition operations by reducing the number of inversions needed.
    
        Args:
        x1, y1 (int, int): Coordinates of the first point.
        x2, y2 (int, int): Coordinates of the second point.
        invx2x1 (int): Precomputed inverse of (x2 - x1) modulo the curve prime P.
    
        Returns:
        tuple[int, int, int, int]: The coordinates of the third and fourth points after the double-add operation.
        """
        # Perform operations as per elliptic curve addition rules using precomputed inverse.
        p = self.P
        dy = (y2 - y1)
        a = dy * invx2x1 % p
        a2 = a**2
        x3 = (a2 - x1 - x2) % p
        y3 = (a * (x1 - x3) - y1) % p

        # Second point addition with modified signs for y-coordinates.
        dy = p - (y2 + y1)
        a = dy * invx2x1 % p
        a2 = a**2
        x4 = (a2 - x1 - x2) % p
        y4 = (a * (x1 - x4) - y1) % p
        return x3, y3, x4, y4


    def b58encode(self, b: bytes) -> str:
        """
        Encodes bytes into a Base58 encoded string.
    
        Args:
        b (bytes): Byte sequence to encode.
    
        Returns:
        str: A Base58 encoded string.
        """
        # Convert byte sequence to a number.
        n = int.from_bytes(b, 'big')
        chars = deque()
        # Compute Base58 encoding by repeatedly dividing the number by 58.
        while n > 0:
            n, remainder = divmod(n, 58)
            chars.appendleft(self.b58_chars[remainder])
        # Handle leading zeros in the byte sequence.
        chars.appendleft('1' * (len(b) - len(b.lstrip(b'\0'))))
        return ''.join(chars)


    def pub_to_addr(self, points):
        """
        Converts public keys derived from elliptic curve points to Denaro addresses.
    
        Args:
        points (tuple[int, int, int, int]): The x and y coordinates of two points.
    
        Returns:
        tuple[str, str]: The two addresses derived from the provided points.
        """
        x1, y1, x2, y2 = points
        # Encode each point into an address format.
        address1 = self.b58encode((42 if y1 % 2 == 0 else 43).to_bytes(1, 'little') + x1.to_bytes(32, 'little'))
        address2 = self.b58encode((42 if y2 % 2 == 0 else 43).to_bytes(1, 'little') + x2.to_bytes(32, 'little'))
        return address1, address2


    def generate_public_key(self, private_key):
        """
        Generates a public key from a given private key.
    
        Args:
        private_key (int): The private key (a scalar).
    
        Returns:
        tuple[int, int]: The elliptic curve point representing the public key.
        """
        # Convert the private key to binary for the point multiplication process.
        bit = list(bin(private_key)[2:])
        point = (0, 0)  # Start with the identity element.
        # Perform scalar multiplication using the private key bits.
        for i in bit:
            point = self.point_double(point)
            if i == '1':
                point = self.point_add(point, (self.GX, self.GY))
        return point
    
    
    def generate_address_from_private_key(self, private_key):
        """
        Generates a Denaro address and its compressed public key from a given private key.
        
        Args:
        private_key (str): The private key in hexadecimal format.
        
        Returns:
        tuple: A tuple containing the generated address and compressed public key.
        """
        # Obtain public point and compressed public key from the private key
        public_point = self.generate_public_key(int(private_key, 16))
        
        # Convert public point to a Denaro address
        address, _ = self.pub_to_addr([public_point[0], public_point[1],public_point[0], public_point[1]])
        compressed_public_key = '02' + format(public_point[0], '064x') if public_point[1] % 2 == 0 else '03' + format(public_point[0], '064x')

        return address, compressed_public_key
    

    def process_key_batches(self, vanity_prefix, silent):
        """
        This method is the primary orchestrator for vanity address generation. It operates by iterating over a range
        of private keys, computing ECC points, and converting these points to Denaro addresses to check against the vanity
        prefix. It utilizes batch processing and efficient ECC point operations to generate potential matches. The process
        is computationally intensive and is designed to run indefinitely until a matching address is found.
    
        Args:
        vanity_prefix (str): The desired prefix for the Denaro address.
        silent (bool): If True, suppresses regular output to streamline processing.
        """
        # Generate a random starting private key from 0 to 2^256.
        lowest_key = secrets.randbelow(2**256)
        start_point = lowest_key + 2048
        distance_between_batches = 4097  # Determines the step size between batches.
    
        # Calculate the number of lines for the progress message, adjust based on 'silent' flag.
        #progress_msg_lines_length = len(self.PROGRESS_MESSAGE_LINES) if not silent else 0
        
        # Display a warning message about the resource-intensive operation.
        self.console_msg = f'WARNING: This operation continuously generates Denaro addresses until a match with the given prefix is found.\nThis may be resource intensive. Press Ctrl+C to exit.\n\nGenerating vanity address: "{vanity_prefix}' + 'x' * (45 - len(vanity_prefix)) + '"'
        
        self.silent = silent
        # Start the progress reporting thread if not in silent mode.
        if not silent:
            self.start_time = time.time()
            self.count = 0

        self.progress_thread = threading.Thread(target=self.update_progress, args=(vanity_prefix,), daemon=True)
        self.progress_thread.start()
    
        # Main loop to process batches starting from the initial private key.
        for k in range(start_point, 2**256, distance_between_batches):
            # Scalar multiplication to find the ECC point corresponding to k.
            kG = self.scalar_multiply(k, (self.GX, self.GY))
            inv_jkz = self.inv(kG[1], self.P)  # Modular inverse of the y-coordinate.
    
            # Compute the batch of inverses for the x-coordinate differences.
            kminverse = self.inv_batch(kG[0], self.mGx[1:], self.P)
            kminverse = [inv_jkz] + kminverse
    
            # Lists to hold constant x and y values for the batch operation.
            kxl = [kG[0]] * 2049
            kyl = [kG[1]] * 2049
    
            # Perform batch point addition and doubling operations.
            batch = [(0,0,0,0)] * 2048
            batch = list(map(self.double_add_P_Q_inv, kxl[1:], kyl[1:], self.mGx[1:], self.mGy[1:], kminverse[1:]))
    
            # Prepend the original point to the batch for processing.
            batch = [(kG[0], kG[1], kG[0], kG[1])] + batch
    
            # Convert ECC points in the batch to Denaro addresses.
            addresses = list(map(self.pub_to_addr, batch))
    
            # Check each generated address for a match with the vanity prefix.
            for i, (address_1, address_2) in enumerate(addresses):
                if not silent:
                    self.count += 2  # Update count of processed addresses.
    
                # Check for prefix match and handle discovery of matching address.
                if address_1.startswith(vanity_prefix) or address_2.startswith(vanity_prefix):
                    private_key_hex = hex(k + i if address_1.startswith(vanity_prefix) else k - i)[2:].rjust(64, '0')
    
                    # Get the relevant ECC point coordinates from the batch based on which address matched.
                    (kmx1, kmy1, kmx2, kmy2) = batch[i]
                    kmx, kmy = (kmx1, kmy1) if address_1.startswith(vanity_prefix) else (kmx2, kmy2)
                    compressed_public_key = '02' + format(kmx, '064x') if kmy % 2 == 0 else '03' + format(kmx, '064x')
    
                    # Output the result and terminate the process.
                    print(f"\nFound matching address!\n\
                            \nPrivate Key: 0x{private_key_hex}\
                            \nPublic Key: 0x{compressed_public_key}\
                            \nAddress: {address_1 if address_1.startswith(vanity_prefix) else address_2}\n")
                    sys.exit(0)


class ArgParseHelpers:
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set the level for the root logger
    root_logger.setLevel(logging.INFO)
    
    # Create a handler with the desired format
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    
    # Clear any existing handlers from the root logger and add our handler
    root_logger.handlers = []
    root_logger.addHandler(handler)


    def check_args(self, args):
        """
        Validates the combinations of CLI arguments specifically for address generation commands.
        It checks the validity of combinations and formats of the provided arguments such as 'private_key', 'prefix'. 
        Errors are reported through logging and the process is terminated if invalid arguments are found.
        
        Parameters:
        - args (argparse.Namespace): The argparse namespace containing parsed arguments.
        """
        # Check if the private_key argument is provided
        if args.private_key:
            context_str = "It must be a hexdecimal string and 64 characters in length."
            
            # Check if prefix argument is used together with private_key
            if args.prefix:
                # If found, sort these arguments for clearer error messages
                sorted_args = self.sort_arguments_based_on_input(['-private_key', '-prefix'])
                # Log error and exit
                logging.error(f"{sorted_args} cannot be used together.")
                sys.exit(1)
    
            # Check the length of the provided private key
            private_key_length = len(args.private_key)
            if private_key_length != 64:
                # Log error for incorrect private key length and exit
                logging.error(f"Private key is too {'long' if private_key_length > 64 else 'short'} ({len(args.private_key)} Characters). {context_str}")
                sys.exit(1)
            
            # Define the regular expression pattern for the private key
            private_key_pattern = r'^(0x)?[0-9a-fA-F]{64}$'
            if not re.match(private_key_pattern, args.private_key):
                # Log error if the private key doesn't match the pattern and exit
                logging.error(f"The provided private key is not valid. {context_str}")
                sys.exit(1)
    
        # Check if the prefix argument is provided
        if args.prefix:
    
            # Check the length of the provided prefix
            vanity_prefix_length = len(args.prefix)
            if vanity_prefix_length > 45:
                # Log error if the prefix length exceeds 45 characters and exit
                logging.error(f"Address prefix is too long ({vanity_prefix_length} Characters). It must be 1 to 45 characters in length.\
                              \nAnything more than 5 characters may take a long time time to generate depending on your hardware.")
                sys.exit(1)
                        
            # Compile the regex pattern for address validation based on the prefix length
            address_pattern = '^-?[DE][1-9A-HJ-NP-Za-km-z]{'+str(vanity_prefix_length-1)+'}$'
            address_pattern = re.compile(address_pattern)
            
            # Validate the vanity prefix
            if not re.match(address_pattern, args.prefix) or not (args.prefix.startswith("D") or args.prefix.startswith("E")):               
                # Highlight and log invalid characters in the prefix if match fails
                highlighted = self.highlight_non_matching_characters(args.prefix, r'^[1-9A-HJ-NP-Za-km-z]$')
                logging.error(f"Invalid characters in address prefix (highlighted in red): {highlighted}")
                if not (args.prefix.startswith("D") or args.prefix.startswith("E")):
                    logging.error('Address prefix must start with "D" or "E".')
                sys.exit(1)
    
    
    def highlight_non_matching_characters(self, text, pattern=None):
        """
        Highlights characters in a string that do not match a given regex pattern.
    
        Args:
        text (str): The text to be scanned.
    
        Returns:
        str: The original text with non-matching characters highlighted in red.
        """
        # This will store the final result
        highlighted_text = ""
    
        # If the initial character does not match D or E, color it red
        if not (text.startswith("D") or text.startswith("E")):
            highlighted_text += f"\033[91m{text[:1]}\033[0m"
        else:
            highlighted_text += text[:1]
    
        # Check each character after the initial character against the pattern
        for char in text[1:]:        
            regex = re.compile(pattern)
            # If the character does not match, color it red
            if not regex.match(char):
                highlighted_text += f"\033[91m{char}\033[0m"
            else:
                highlighted_text += char
    
        return highlighted_text
    
    
    def sort_arguments_based_on_input(argument_names):
        """
        Overview:
            Sorts a list of CLI argument names based on their positional occurrence in sys.argv.
            Any argument not found in sys.argv is filtered out. The returned list is then formatted
            as a comma-separated string. This version also handles arguments with an '=' sign.
    
            Parameters:
            - argument_names (list): A list of argument names to be sorted.
        
            Returns:
            - str: A string of sorted argument names separated by commas with 'and' added before the last argument.
        """
        # Process each argument in sys.argv to extract the argument name before the '=' sign
        processed_argv = [arg.split('=')[0] for arg in sys.argv]
    
        # Filter out arguments that are not present in the processed sys.argv
        filtered_args = [arg for arg in argument_names if arg in processed_argv]
    
        # Sort the filtered arguments based on their index in the processed sys.argv
        sorted_args = sorted(filtered_args, key=lambda x: processed_argv.index(x))    
    
        # Join the arguments into a string with proper formatting
        if len(sorted_args) > 1:
            result = ', '.join(sorted_args[:-1]) + ', and ' + sorted_args[-1]    
        elif sorted_args:
            result = sorted_args[0]
        else:
            result = ''
        return result
    

# Overview:
# This serves as the main entry point for the command-line vanity address generator.
# It supports generating addresses either by providing a private key directly or by specifying
# a prefix for a vanity address. It includes argument validation and can operate in silent mode
# to suppress progress output.
def main():    
    vanityGen = VanityGen()
    argParseHelpers = ArgParseHelpers()
    parser = argparse.ArgumentParser(description="A vanity address generator for the Denaro cryptocurrency. Developed by The-Sycorax.")
    
    # CLI arguments
    parser.add_argument("-prefix", type=str, help="The prefix of the vanity address to generate.")
    parser.add_argument("-silent", help="Suppresses progress output when generating a vanity address.", action='store_true')
    parser.add_argument("-private_key", type=str, help="A private key to generate a single Denaro address. Must be a hexadecimal private key and 64 characters in length. This argument is separate from vanity address generation and should be used independently.")
    
    # Main parser
    args = parser.parse_args()
    
    # Check if the private_key argument is provided
    if args.private_key:
        #Remove '0x' prefix from private key if present
        args.private_key = args.private_key.replace('0x','')
        # Validate arguments
        argParseHelpers.check_args(args)
        # Generate address from the provided private key
        address, compressed_pub_key = vanityGen.generate_address_from_private_key(args.private_key)
        # Output the results
        print(f"\nPrivate Key: 0x{args.private_key}\nPublic Key: 0x{compressed_pub_key}\nAddress: {address}\n")

    # Check if the prefix argument for vanity address generation is provided
    if args.prefix:
        # Validate arguments specific for vanity generation
        argParseHelpers.check_args(args)
        vanityGen.process_key_batches(args.prefix, args.silent if args.silent else False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\r  ")
        print("\rProcess terminated by user.")
    except Exception as e:
        print("\r  ")
        # Print the error message
        print(f"\r\033[91mGuru Meditation Error:\033[0m\n{e}")
        # Print the traceback
        traceback.print_exc()
    sys.exit(0)