from kafka import KafkaProducer
import json
import time
import random

# Create a producer that serializes data to JSON
def create_producer():
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda v: str(v).encode('utf-8'),
        acks='all',
        retries=3,
        linger_ms=10
    )
    return producer

# Function to generate sample data
def generate_data():
    return {
        'user_id': random.randint(1, 1000),
        'page_id': random.randint(1, 100),
        'visit_time': time.time(),
        'duration': random.randint(1, 300),
        'actions': random.randint(0, 10)
    }

# Send data to Kafka
def send_data(producer, topic, data, key=None):
    if key is None and 'user_id' in data:
        key = data['user_id']
    
    future = producer.send(topic, value=data, key=key)
    
    try:
        record_metadata = future.get(timeout=10)
        print(f"Message sent successfully to {record_metadata.topic} "
              f"[partition {record_metadata.partition}] @ offset {record_metadata.offset}")
        return record_metadata
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

# Example usage
def run_producer(topic_name, num_messages=10, delay=1):
    producer = create_producer()
    
    try:
        for i in range(num_messages):
            data = generate_data()
            send_data(producer, topic_name, data)
            time.sleep(delay)
    finally:
        producer.flush()
        producer.close()
        print("Producer closed")

if __name__ == "__main__":
    run_producer("website_visits", 10, 0.5)
