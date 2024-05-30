# Denaro Vanity Generator

## Introduction
**This repo contains the source code for the Denaro Vanity Generator, a vanity address generator developed for the Denaro cryptocurrency. It features a highly efficient algorithm capable of producing tens of thousands of Denaro addresses per second. Actual performance will depend on your hardware capabilities.**

The original generator algorithm derives from a set of Python scripts posted on the Bitcointalk forum. This algorithm was designed to rapidly generate up to 16 million Bitcoin addresses in under 30 seconds due to its use of secp256k1 elliptic curve endomorphisms. Since Denaro requires the P256 (secp256r1) elliptic curve to generate valid public keys for address derivation, the implementation has been modified to use P256 accordingly.

The P256 curve unfortunately does not support endomorphisms, which the original algorithm uses. As a result, the modified generator cannot achieve the same high throughput, significantly reducing its ability to produce millions of keys quickly. Regardless of this limitation, the use of P256 still ensures strong security standards and offers the potential for integrating alternative optimizations in future updates to this project.

* **Original source: https://bitcointalk.org/index.php?topic=5432068.msg61517943#msg61517943** 

* **Other links:**

    - **Denaro Cryptocurrency: https://github.com/denaro-coin/denaro**
    
    - **Denaro Wallet Client: https://github.com/The-Sycorax/DenaroWalletClient**



## Installation
*Note: The Denaro Vanity Generator has not been tested on Windows or MacOS and support is unknown at this time. It is reccomended to run it on Ubuntu/Debian Linux to avoid any compatibility or stability issues.*

```bash
# Clone the repository
git clone https://github.com/The-Sycorax/Denaro-Vanity-Generator.git
cd Denaro-Vanity-Generator

# Install the required packages
pip3 install humanfriendly

# Run the vanity generator
python3 vanity_gen.py <options>
```

## Usage
### Syntax:
```bash
vanity_gen_v2.py [-h] [-prefix PREFIX] [-private_key PRIVATE_KEY] [-silent]
```
    
- **Options**:        
    * `-prefix`: The prefix of the vanity address to generate.
    * `-private_key`: A private key to generate a single Denaro address. Must be a hexadecimal private key and 64 characters in length. This argument is separate from vanity address generation and should be used independently. 
    
    * `-silent`: Suppresses progress output when generating a vanity address.

### **Example**:
```bash
python3 vanity_gen_v2.py -prefix=Denaro
```

------------

## Disclaimer

Neither The-Sycorax nor contributors of this project assume liability for any loss of funds incurred through the use of this software! This software is provided 'as is' under the [MIT License](LICENSE) without guarantees or warrenties of any kind, express or implied. It is strongly recommended that users back up their cryptographic keys. User are solely responsible for the security and management of their assets! The use of this software implies acceptance of all associated risks, including financial losses, with no liability on The-Sycorax or contributors of this project.

------------

## License
The Denaro Vanity Generator is released under the terms of the MIT license. See [LICENSE](LICENSE) for more
information or see https://opensource.org/licenses/MIT.
