"""Mock script to produce messages to Kafka for testing."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluent_kafka import Producer
from config import settings

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_test_messages():
    conf = {'bootstrap.servers': settings.kafka_brokers}
    producer = Producer(conf)

    messages = [
        {
            "topic": "teams-messages",
            "data": {
                "text": "The new AT&T network architecture reduced latency by 30% in Q1.",
                "doc_title": "Network Report",
                "source_name": "Engineering Team Channel",
                "url": "teams://channel/1",
                "source_type": "Teams Message"
            }
        },
        {
            "topic": "raw-emails",
            "data": {
                "text": "Q1 showed a 30% drop in latency due to the AT&T network upgrades.",
                "doc_title": "Q1 Summary Email",
                "source_name": "Management Thread",
                "url": "email://thread/1",
                "source_type": "Email"
            }
        }
    ]

    for m in messages:
        producer.produce(
            m["topic"],
            json.dumps(m["data"]).encode('utf-8'),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(1)

    producer.flush()

if __name__ == "__main__":
    produce_test_messages()
