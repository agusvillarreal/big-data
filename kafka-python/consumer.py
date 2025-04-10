from kafka import KafkaConsumer
import json
import time

# Create a consumer that deserializes JSON data
def create_consumer(topic, group_id=None):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        key_deserializer=lambda m: m.decode('utf-8') if m else None
    )
    return consumer

# Process messages from Kafka
def process_message(message):
    topic = message.topic
    partition = message.partition
    offset = message.offset
    key = message.key
    value = message.value
    
    print(f"Received message from {topic} [partition {partition}] @ offset {offset}")
    print(f"Key: {key}")
    print(f"Value: {value}")
    print(f"Timestamp: {message.timestamp}")
    print("-" * 50)
    
    if 'duration' in value and 'actions' in value:
        engagement = value['actions'] / value['duration'] if value['duration'] > 0 else 0
        print(f"User {value.get('user_id')} engagement score: {engagement:.4f}")
    
    return True

# Run the consumer loop
def run_consumer(topic_name, group_id=None, timeout_ms=10000, max_messages=None):
    consumer = create_consumer(topic_name, group_id)
    
    try:
        message_count = 0
        
        for message in consumer:
            success = process_message(message)
            
            if success:
                message_count += 1
            
            if max_messages is not None and message_count >= max_messages:
                print(f"Reached maximum message count ({max_messages}). Stopping.")
                break
                
    except KeyboardInterrupt:
        print("Consumer stopped by user")
    finally:
        consumer.close()
        print("Consumer closed")

if __name__ == "__main__":
    run_consumer("website_visits", "website_analytics_group", max_messages=10)
