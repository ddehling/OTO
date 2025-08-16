import socket
import struct
import time
import subprocess
import sys

def test_pixlite_direct():
    """
    Direct connection test to known PixLite IP
    """
    pixlite_ip = "192.168.68.121"
    
    print("=" * 60)
    print(f"Direct PixLite Connection Test to {pixlite_ip}")
    print("=" * 60)
    
    # First, check if we can reach the IP at all
    print(f"\n1. Testing basic connectivity to {pixlite_ip}...")
    
    # Ping test
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['ping', '-n', '2', pixlite_ip], 
                                  capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run(['ping', '-c', '2', pixlite_ip], 
                                  capture_output=True, text=True, timeout=5)
        
        if "TTL=" in result.stdout or "ttl=" in result.stdout:
            print(f"   ✓ Ping successful - device is reachable")
        else:
            print(f"   ✗ Ping failed - device may be offline or blocking ICMP")
            print(f"   Output: {result.stdout[:200]}")
    except Exception as e:
        print(f"   ✗ Ping error: {e}")
    
    # Check what ports are actually open
    print(f"\n2. Scanning all common ports on {pixlite_ip}...")
    
    common_ports = {
        21: "FTP",
        22: "SSH", 
        23: "Telnet",
        80: "HTTP",
        443: "HTTPS",
        502: "Modbus",
        1900: "UPnP",
        4048: "PixLite Discovery",  # Some models use this
        5568: "sACN/E1.31",
        6454: "Art-Net",
        9930: "Control",
        30000: "PixLite",
        49150: "PixLite Web",
        49151: "PixLite Alt",
        49152: "PixLite Config",
        50000: "PixLite Data",
        50001: "PixLite Sync"
    }
    
    open_ports = []
    
    for port, service in common_ports.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((pixlite_ip, port))
            sock.close()
            
            if result == 0:
                print(f"   ✓ Port {port:5} ({service}) - OPEN")
                open_ports.append(port)
            else:
                # Also try UDP for certain services
                if port in [5568, 6454, 4048, 30000]:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.settimeout(0.5)
                        sock.sendto(b"test", (pixlite_ip, port))
                        sock.close()
                        print(f"   ? Port {port:5} ({service}) - UDP (no error)")
                    except:
                        pass
        except Exception as e:
            print(f"   ✗ Port {port:5} ({service}) - Error: {e}")
    
    # Test sACN specifically
    print(f"\n3. Testing E1.31/sACN multicast...")
    
    try:
        # Create proper E1.31 packet
        universe = 1
        
        # Root Layer
        packet = bytearray(638)
        packet[0:14] = b'\x00\x10\x00\x00ASC-E1.31\x00'
        packet[14:16] = struct.pack('!H', 0x7000 | 638)  # Flags and length
        packet[16:20] = b'\x00\x00\x00\x00'  # Vector
        packet[20:36] = b'PixLite Test\x00\x00\x00\x00'  # CID
        
        # Framing Layer
        packet[38:40] = struct.pack('!H', 0x7000 | 600)
        packet[40:44] = struct.pack('!I', 0x00000002)  # Vector
        packet[44:108] = b'PixLite Discovery Test' + b'\x00' * 42  # Source name
        packet[108] = 100  # Priority
        packet[109:111] = struct.pack('!H', 0)  # Reserved
        packet[111] = 0  # Sequence
        packet[112] = 0  # Options
        packet[113:115] = struct.pack('!H', universe)  # Universe
        
        # DMP Layer  
        packet[115:117] = struct.pack('!H', 0x7000 | 523)
        packet[117] = 0x02  # Vector
        packet[118] = 0xa1  # Address Type & Data Type
        packet[119:121] = struct.pack('!H', 0x0000)  # First Address
        packet[121:123] = struct.pack('!H', 0x0001)  # Address Increment
        packet[123:125] = struct.pack('!H', 513)  # Property value count
        packet[125] = 0  # START Code
        
        # Send to multicast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 20)
        
        multicast_ip = f"239.255.0.{universe}"
        sock.sendto(packet, (multicast_ip, 5568))
        print(f"   ✓ Sent sACN to multicast {multicast_ip}:5568")
        
        # Also send unicast directly
        sock.sendto(packet, (pixlite_ip, 5568))
        print(f"   ✓ Sent sACN unicast to {pixlite_ip}:5568")
        
        sock.close()
        
    except Exception as e:
        print(f"   ✗ sACN test error: {e}")
    
    # Try Art-Net
    print(f"\n4. Testing Art-Net...")
    
    try:
        # Art-Net ArtPoll packet
        artpoll = bytearray([
            0x41, 0x72, 0x74, 0x2d, 0x4e, 0x65, 0x74, 0x00,  # "Art-Net\0"
            0x00, 0x20,  # OpPoll
            0x00, 0x0e,  # Protocol version
            0x02, 0x00   # TalkToMe (2 = unicast response)
        ])
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        
        # Send directly to PixLite
        sock.sendto(artpoll, (pixlite_ip, 6454))
        print(f"   ✓ Sent Art-Net poll to {pixlite_ip}:6454")
        
        # Try to receive response
        try:
            data, addr = sock.recvfrom(1024)
            if data[:8] == b'Art-Net\x00':
                print(f"   ✓ Received Art-Net response from {addr[0]}:{addr[1]}")
                # Parse some basic info
                if len(data) > 18:
                    opcode = struct.unpack('<H', data[8:10])[0]
                    print(f"     OpCode: 0x{opcode:04x}")
        except socket.timeout:
            print(f"   - No Art-Net response received")
            
        sock.close()
        
    except Exception as e:
        print(f"   ✗ Art-Net test error: {e}")
    
    # Check network configuration
    print(f"\n5. Network Configuration Check...")
    
    try:
        import ipaddress
        
        # Get local IPs
        hostname = socket.gethostname()
        local_ips = socket.gethostbyname_ex(hostname)[2]
        
        pixlite_net = ipaddress.ip_network(f"{pixlite_ip}/24", strict=False)
        
        on_same_network = False
        for local_ip in local_ips:
            local_net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
            print(f"   Local IP: {local_ip}")
            if local_net == pixlite_net:
                print(f"     ✓ On same network as PixLite")
                on_same_network = True
            else:
                print(f"     - Different network from PixLite")
        
        if not on_same_network:
            print(f"\n   ⚠ WARNING: No local IP on same network as PixLite!")
            print(f"   PixLite is on: {pixlite_net}")
            print(f"   This may prevent discovery/communication")
            
    except Exception as e:
        print(f"   Error checking network: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    if open_ports:
        print(f"✓ Found {len(open_ports)} open ports on {pixlite_ip}")
        print(f"  Open ports: {open_ports}")
    else:
        print(f"✗ No open ports found on {pixlite_ip}")
        print("  Possible issues:")
        print("  - Wrong IP address")
        print("  - Firewall blocking connections")
        print("  - PixLite on different VLAN/network segment")
        print("  - PixLite in different mode (check if in setup mode)")
    
    print("\nRECOMMENDATIONS:")
    print("1. Check PixLite software connection settings")
    print("2. Note which port/protocol the software uses")
    print("3. Check if PixLite has security/firewall settings enabled")
    print("4. Verify network adapter settings match PixLite network")
    print("5. Try temporarily disabling Windows Firewall")
    
    return open_ports

def capture_pixlite_traffic():
    """
    Monitor network traffic to see what the PixLite software uses
    """
    print("\n" + "=" * 60)
    print("Network Traffic Monitor")
    print("=" * 60)
    print("\nThis will show what protocols are being used...")
    print("Start your PixLite software and connect to the device.\n")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
        
        # Bind to local interface
        host = socket.gethostbyname(socket.gethostname())
        sock.bind((host, 0))
        
        # Include IP headers
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        
        # Enable promiscuous mode on Windows
        if sys.platform == 'win32':
            sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
        
        print(f"Monitoring traffic (press Ctrl+C to stop)...")
        print(f"Looking for traffic to/from 192.168.68.121\n")
        
        pixlite_ip = "192.168.68.121"
        seen_ports = set()
        
        while True:
            data, addr = sock.recvfrom(65535)
            
            # Parse IP header
            if len(data) >= 20:
                # Get source and dest IPs
                src_ip = socket.inet_ntoa(data[12:16])
                dst_ip = socket.inet_ntoa(data[16:20])
                
                # Check if involves PixLite
                if pixlite_ip in [src_ip, dst_ip]:
                    protocol = data[9]
                    
                    if protocol == 6:  # TCP
                        if len(data) >= 40:
                            src_port = struct.unpack('!H', data[20:22])[0]
                            dst_port = struct.unpack('!H', data[22:24])[0]
                            
                            port_key = f"TCP:{src_port}->{dst_port}"
                            if port_key not in seen_ports:
                                seen_ports.add(port_key)
                                print(f"TCP: {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
                                
                    elif protocol == 17:  # UDP
                        if len(data) >= 28:
                            src_port = struct.unpack('!H', data[20:22])[0]
                            dst_port = struct.unpack('!H', data[22:24])[0]
                            
                            port_key = f"UDP:{src_port}->{dst_port}"
                            if port_key not in seen_ports:
                                seen_ports.add(port_key)
                                print(f"UDP: {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
                                
    except PermissionError:
        print("✗ Permission denied - run as Administrator")
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    except Exception as e:
        print(f"✗ Monitor error: {e}")
        print("Note: This requires Administrator/root privileges")
    finally:
        if sys.platform == 'win32' and 'sock' in locals():
            sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)

def main():
    # First do direct test
    open_ports = test_pixlite_direct()
    
    # Offer to monitor traffic
    if sys.platform == 'win32':
        print("\n" + "=" * 60)
        response = input("\nMonitor network traffic to detect PixLite protocol? (requires Admin) (y/n): ")
        if response.lower() == 'y':
            capture_pixlite_traffic()

if __name__ == "__main__":
    main()