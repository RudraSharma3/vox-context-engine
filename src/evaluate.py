# src/evaluate.py
import requests
import json
import time
import sys

API_URL = "http://127.0.0.1:8000/query"

BENCHMARK_QUESTIONS = [
    "What pricing plan includes API access?",
    "How does the system handle high-throughput file ingestion processing constraints?",
    "What are the default configuration values for chunk sizes and overlap thresholds?",
    "Does the system support secure data compliance rules and multi-tenancy token isolation?",
    "What external database engines are used to store indexing payload structures?"
]

def run_evaluation_suite():
    print("==========================================================", flush=True)
    print("      RUNNING RETRIEVAL & RESPONSE EVALUATION SUITE       ", flush=True)
    print("==========================================================\n", flush=True)
    
    successful_runs = 0
    
    for idx, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"Test Case #{idx}: '{question}'", flush=True)
        payload = {"question": question}
        
        # Add a short delay between iterations to handle free-tier API quotas safely
        if idx > 1:
            time.sleep(4)
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"  [STATUS]  Success (200 OK)", flush=True)
                print(f"  [LATENCY] {latency:.3f} seconds", flush=True)
                print(f"  [SOURCES] {data.get('sources', [])}", flush=True)
                print(f"  [ANSWER]  {data.get('answer', '')}", flush=True)
                successful_runs += 1
            else:
                print(f"  [STATUS]  Failed with Status Code: {response.status_code}", flush=True)
                print(f"  [ERROR]   {response.text}", flush=True)
                
        except requests.exceptions.ConnectionError:
            print("  [CRITICAL] Could not connect to the local server. Make sure your FastAPI app is running on port 8000!", flush=True)
            break
        except requests.exceptions.Timeout:
            print("  [TIMEOUT]  The upstream connection timed out after 15 seconds.", flush=True)
        except Exception as e:
            print(f"  [ERROR]   An unexpected anomaly occurred: {str(e)}", flush=True)
            
        print("-" * 58 + "\n", flush=True)
        
    print("==========================================================", flush=True)
    print(f" Evaluation Complete: {successful_runs}/{len(BENCHMARK_QUESTIONS)} Test Cases Processed Safely.", flush=True)
    print("==========================================================", flush=True)

if __name__ == "__main__":
    run_evaluation_suite()