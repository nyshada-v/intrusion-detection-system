import scapy.all as scapy

from scapy.layers.inet import IP, TCP, UDP

import threading

import time

import pandas as pd

import numpy as np
# Add an on_stats_update callback to your __init__
def __init__(self, on_flow_complete, on_stats_update=None, interface=None, bpf_filter="ip"):
    self.on_stats_update = on_stats_update # <--- New callback
    # ... rest of your init ...

def _process_packet(self, pkt):
    if IP not in pkt:
        return

    self.packets_captured += 1
    
    # Send live stats to the UI immediately
    if self.on_stats_update:
        self.on_stats_update(self.stats())
    


def list_interfaces():

    """Returns a list of available network interfaces for the UI."""

    return [str(iface) for iface in scapy.get_if_list()]



class PacketCapture:

    def __init__(self, on_flow_complete, interface=None, bpf_filter="ip"):

        self.on_flow_complete = on_flow_complete

        self.interface = interface

        self.filter = bpf_filter

        self.running = False

        self.thread = None

        

        # Stats & Flow Tracking

        self.packets_captured = 0

        self.flows = {} # Key: 5-tuple, Value: Running stats

        self.flow_timeout = 60  # Seconds of inactivity before forcing a prediction



    def start(self):

        self.running = True

        self.thread = threading.Thread(target=self._run_sniff, daemon=True)

        self.thread.start()

        print(f"[Capture] Started sniffing on {self.interface or 'default'}")



    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(timeout=1)



    def stats(self):

        return {

            "packets_captured": self.packets_captured,

            "active_flows": len(self.flows)

        }



    def _run_sniff(self):

        scapy.sniff(

            iface=self.interface,

            filter=self.filter,

            prn=self._process_packet,

            stop_filter=lambda x: not self.running,

            store=0  # KEY DIFFERENCE: Discards raw packet after processing

        )



    def _process_packet(self, pkt):

        if IP not in pkt:

            return



        self.packets_captured += 1

        

        # 1. Extract 5-tuple

        proto = pkt.proto

        src_ip = pkt[IP].src

        dst_ip = pkt[IP].dst

        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)

        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

        flow_id = (src_ip, dst_ip, sport, dport, proto)



        now = time.time()



        # 2. Update Stats

        if flow_id not in self.flows:

            self.flows[flow_id] = self._init_flow_stats(pkt, now)

        else:

            self._update_flow_stats(flow_id, pkt, now)



        # 3. Check if flow should end (e.g., TCP FIN/RST or timeout)

        if self._should_terminate(pkt, flow_id):

            features = self._finalize_flow(flow_id)

            # Pass features to inference.py via the callback in main.py

            self.on_flow_complete(features)

            del self.flows[flow_id]



    def _init_flow_stats(self, pkt, timestamp):

        return {

            "first_timestamp": timestamp,

            "last_timestamp": timestamp,

            "Total Fwd Packets": 1,

            "Total Length of Fwd Packets": len(pkt),

            "Total Backward Packets": 0,

            "Total Length of Bwd Packets": 0,

            "Flow Duration": 0,

            "Flow Bytes/s": 0,

            "Flow Packets/s": 0,

            "Packet Length Mean": len(pkt),

            "Packet Length Std": 0,

            # Add other CICIDS features your model needs here

        }



    def _update_flow_stats(self, flow_id, pkt, timestamp):

        flow = self.flows[flow_id]

        flow["last_timestamp"] = timestamp

        flow["Total Fwd Packets"] += 1

        flow["Total Length of Fwd Packets"] += len(pkt)

        flow["Flow Duration"] = (timestamp - flow["first_timestamp"]) * 1e6 # microseconds



    def _should_terminate(self, pkt, flow_id):

        # Terminate on TCP FIN or RST flags

        if TCP in pkt:

            flags = pkt[TCP].underlying_attrs.get('flags', 0)

            if flags & 0x01 or flags & 0x04: # FIN or RST

                return True

        return False



    def _finalize_flow(self, flow_id):

        # Convert the internal dict into the feature set inference.py expects

        raw_stats = self.flows[flow_id]

        # Basic mapping to match your inference.py DROP_COLS and feature engineering

        return {

            "Flow Duration": raw_stats["Flow Duration"],

            "Total Fwd Packets": raw_stats["Total Fwd Packets"],

            "Total Backward Packets": raw_stats["Total Backward Packets"],

            "Total Length of Fwd Packets": raw_stats["Total Length of Fwd Packets"],

            "Total Length of Bwd Packets": raw_stats["Total Length of Bwd Packets"],

            "Packet Length Mean": raw_stats["Packet Length Mean"],

            "Packet Length Std": raw_stats["Packet Length Std"],

            # Metadata for UI

            "Source IP": flow_id[0],

            "Destination IP": flow_id[1],

            "Source Port": flow_id[2],

            "Destination Port": flow_id[3]

        }