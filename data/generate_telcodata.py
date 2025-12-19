import csv
import json
import random
import uuid
import time
import os
import ipaddress
from datetime import datetime, timedelta

# Configuration
DATA_DIR = "data"
SIP_DIR = os.path.join(DATA_DIR, "sip")
GTPU_DIR = os.path.join(DATA_DIR, "gtpu")
CDR_DIR = os.path.join(DATA_DIR, "cdr")

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

def ensure_dirs():
    for d in [SIP_DIR, GTPU_DIR, CDR_DIR]:
        os.makedirs(d, exist_ok=True)

# --- Utilities ---
def random_ip():
    return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

def random_imei():
    # 8 digits TAC + 6 digits SN + 1 check digit (ignored)
    tac = random.choice([
        "35209900", # Samsung fake
        "86429000", # Apple fake
        "99000010"  # IoT Module fake
    ])
    return f"{tac}{random.randint(100000, 999999)}"

def random_imsi():
    # MCC (3) + MNC (2-3) + MSIN
    return f"310410{random.randint(1000000000, 9999999999)}"

def random_msisdn():
    return f"1{random.randint(2000000000, 9999999999)}"

# --- SIP Generator ---
SIP_METHODS = ["INVITE", "ACK", "BYE", "REGISTER", "OPTIONS", "CANCEL", "PRACK"]
SIP_STATUS = {
    "INVITE": [100, 180, 200, 486, 404, 503],
    "REGISTER": [200, 401, 403],
    "BYE": [200],
    "OPTIONS": [200]
}
USER_AGENTS = [
    "Asterisk PBX 18.2", "Cisco-CP-8841/14.0", "Avaya IP Phone 9608", 
    "Linphone_Android_4.2", "MicroSIP/3.19.3", "Python-Sip-Attacker-v1"
]

def generate_sip_data(num_calls=1000):
    print(f"Generating {num_calls} SIP call flows...")
    filepath = os.path.join(SIP_DIR, f"sip_trace_{int(time.time())}.jsonl")
    
    with open(filepath, 'w') as f:
        for _ in range(num_calls):
            call_id = str(uuid.uuid4())
            src_ip = random_ip()
            user_agent = random.choice(USER_AGENTS)
            
            # Scenario 1: Successful Call (Standard Flow)
            if random.random() < 0.7:
                flow = ["INVITE", "100", "180", "200", "ACK", "BYE", "200"]
                
            # Scenario 2: Busy / Rejected
            elif random.random() < 0.9:
                flow = ["INVITE", "100", "486"]
                
            # Scenario 3: Network Failure (MOS impact)
            else:
                flow = ["INVITE", "100", "503"]
                
            # Scenario 4: Attack (INSERTION) - High volume handled by main loop multiplier usually, 
            # but let's make a specific "flood" burst for a specific IP later.
            
            base_time = datetime.now() - timedelta(minutes=random.randint(0, 60))
            
            for step in flow:
                base_time += timedelta(milliseconds=random.randint(10, 200))
                
                is_response = step.isdigit()
                method = "RESPONSE" if is_response else step
                status = int(step) if is_response else None
                
                record = {
                    "timestamp": base_time.strftime(TIMESTAMP_FORMAT),
                    "call_id": call_id,
                    "method": method if not is_response else None,
                    "status_code": status,
                    "user_agent": user_agent,
                    "contact_uri": f"sip:user@{src_ip}:5060",
                    "sdp_info": "v=0\r\no=- 123 123 IN IP4..." if method == "INVITE" or status == 200 else None
                }
                f.write(json.dumps(record) + "\n")

    # Generate SIP Flood Attack Snippet
    print("Injecting SIP Flood Attack...")
    attacker_ip = "192.168.1.66" # Obvious pattern
    with open(filepath, 'a') as f:
        for _ in range(500):
            record = {
                "timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
                "call_id": str(uuid.uuid4()),
                "method": "REGISTER",
                "status_code": None,
                "user_agent": "Python-Sip-Attacker-v1",
                "contact_uri": f"sip:attacker@{attacker_ip}:5060",
                "sdp_info": None
            }
            f.write(json.dumps(record) + "\n")

# --- GTP-U Generator ---
def generate_gtpu_data(num_sessions=100):
    print(f"Generating GTP-U traffic for {num_sessions} sessions...")
    filepath = os.path.join(GTPU_DIR, f"gtpu_traffic_{int(time.time())}.csv")
    
    headers = ["timestamp", "teid", "inner_src_ip", "inner_dest_port", "payload_length", "sequence_number"]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for _ in range(num_sessions):
            teid = random.randint(100000, 999999)
            inner_ip = random_ip()
            
            # Decide Type: M2M (Machine) vs Human
            if random.random() < 0.3:
                # M2M: Rigid, small, periodic
                # e.g., Smart Meter reporting every 1 sec
                profile = "M2M"
                base_payload = 120
                variance = 5 # Very low variance
                dest_port = 1883 # MQTT
                packet_count = 50
            else:
                # Human: Bursty, large, random
                # e.g., Video streaming or browsing
                profile = "HUMAN"
                base_payload = 1200
                variance = 800 # High variance
                dest_port = 443 # HTTPS
                packet_count = 50
                
            seq = 0
            base_time = datetime.now() - timedelta(minutes=10)
            
            for i in range(packet_count):
                seq += 1
                
                # Payload Logic
                if profile == "M2M":
                    # Deterministic
                    payload = base_payload + random.randint(-variance, variance)
                    time_step = 1.0 # Rigid timing
                else: 
                    # Bursty
                    payload = max(40, base_payload + random.randint(-variance, variance))
                    if random.random() < 0.4: # Silence period
                        time_step = random.uniform(0.1, 5.0)
                        payload = 50 # Keep-alive
                    else: # Burst
                        time_step = random.uniform(0.01, 0.1)
                
                base_time += timedelta(seconds=time_step)
                
                writer.writerow([
                    base_time.strftime(TIMESTAMP_FORMAT),
                    teid,
                    inner_ip,
                    dest_port,
                    payload,
                    seq
                ])

# --- CDR Generator ---
def generate_cdr_data(num_records=5000):
    print(f"Generating {num_records} CDR records...")
    filepath = os.path.join(CDR_DIR, f"cdr_logs_{int(time.time())}.csv")
    
    headers = [
        "timestamp", "imsi", "imei_tac", "msisdn", "rat_type", 
        "uplink_bytes", "downlink_bytes", "duration_sec", 
        "cause_closing", "first_cell_id", "last_cell_id"
    ]
    
    # Create a subscriber pool to simulate churn trends
    subscribers = []
    for _ in range(200): # 200 unique users
        subscribers.append({
            "imsi": random_imsi(),
            "imei": random_imei()[:8],
            "msisdn": random_msisdn(),
            "churn_risk": random.random() < 0.2 # 20% are dropping usage
        })
        
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        start_date = datetime.now() - timedelta(days=90) # 3 months of data
        
        for _ in range(num_records):
            sub = random.choice(subscribers)
            
            # Time distribution
            session_time = start_date + timedelta(days=random.randint(0, 90))
            
            # Usage Logic
            if sub["churn_risk"]:
                # If churning, usage drops in the last 30 days
                days_from_start = (session_time - start_date).days
                if days_from_start > 60:
                    multiplier = 0.1 # Significant drop
                else:
                    multiplier = 1.0
            else:
                multiplier = 1.0
                
            dl_vol = int(random.lognormvariate(10, 2) * multiplier) # Heavy tail
            ul_vol = int(dl_vol * 0.1)
            duration = int(dl_vol / 10000) + random.randint(10, 60)
            
            # Location
            cell_a = random.randint(1000, 5000)
            cell_b = cell_a if random.random() < 0.6 else random.randint(1000, 5000) # 40% mobility
            
            rat = random.choice(["4G", "5G", "Wi-Fi"])
            cause = "Normal Release" if random.random() < 0.95 else "Radio Link Failure"
            
            writer.writerow([
                session_time.strftime(TIMESTAMP_FORMAT),
                sub["imsi"],
                sub["imei"],
                sub["msisdn"],
                rat,
                ul_vol,
                dl_vol,
                duration,
                cause,
                cell_a,
                cell_b
            ])

def main():
    ensure_dirs()
    random.seed(42) # Reproducible
    
    generate_sip_data(num_calls=200)
    generate_gtpu_data(num_sessions=50) # 50 users * 50 packets
    generate_cdr_data(num_records=1000)
    
    print(f"Data generation complete. Check {DATA_DIR}/")

if __name__ == "__main__":
    main()
