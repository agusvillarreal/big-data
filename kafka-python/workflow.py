import threading
import time
import json
import random
from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError
from kafka import KafkaProducer, KafkaConsumer

# Create a Kafka topic
def create_topic(topic_name, num_partitions=3, replication_factor=1):
    admin_client = KafkaAdminClient(bootstrap_servers=['localhost:9092'])
    
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor
    )
    
    try:
        admin_client.create_topics([topic])
        print(f"Topic '{topic_name}' created successfully with {num_partitions} partitions")
    except TopicAlreadyExistsError:
        print(f"Topic '{topic_name}' already exists")
    finally:
        admin_client.close()

# Producer function
def producer_task(topic_name, num_messages, delay=0.5):
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda v: str(v).encode('utf-8')
    )
    
    try:
        for i in range(num_messages):
            data = {
                'user_id': random.randint(1, 1000),
                'page_id': random.randint(1, 100),
                'visit_time': time.time(),
                'duration': random.randint(1, 300),
                'actions': random.randint(0, 10)
            }
            
            key = data['user_id']
            
            future = producer.send(topic_name, value=data, key=key)
            
            record_metadata = future.get(timeout=10)
            
            print(f"[Producer] Sent message {i+1}/{num_messages} to {topic_name} "
                  f"[partition {record_metadata.partition}] @ offset {record_metadata.offset}")
            
            time.sleep(delay)
            
    except Exception as e:
        print(f"[Producer] Error: {e}")
    finally:
        producer.flush()
        producer.close()
        print("[Producer] Closed")

# Consumer function
def consumer_task(topic_name, group_id, consumer_id, max_messages=None):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        key_deserializer=lambda m: m.decode('utf-8') if m else None
    )
    
    try:
        message_count = 0
        for message in consumer:
            user_id = message.value.get('user_id')
            duration = message.value.get('duration', 0)
            actions = message.value.get('actions', 0)
            
            engagement = actions / duration if duration > 0 else 0
            
            print(f"[Consumer {consumer_id}] Received message from {message.topic} "
                  f"[partition {message.partition}] @ offset {message.offset}")
            print(f"[Consumer {consumer_id}] User {user_id} engagement: {engagement:.4f}")
            print("-" * 50)
            
            message_count += 1
            
            if max_messages is not None and message_count >= max_messages:
                print(f"[Consumer {consumer_id}] Processed {message_count} messages. Stopping.")
                break
    
    except KeyboardInterrupt:
        print(f"[Consumer {consumer_id}] Stopped by user")
    finally:
        consumer.close()
        print(f"[Consumer {consumer_id}] Closed")

# Main workflow function
def run_kafka_workflow():
    topic_name = "website_visits"
    
    create_topic(topic_name, num_partitions=3)
    
    consumer_threads = []
    for i in range(3):
        consumer_thread = threading.Thread(
            target=consumer_task,
            args=(topic_name, "website_analytics_group", i, 5)
        )
        consumer_threads.append(consumer_thread)
        consumer_thread.start()
    
    time.sleep(2)
    producer_thread = threading.Thread(
        target=producer_task,
        args=(topic_name, 15, 0.5)
    )
    producer_thread.start()
    
    producer_thread.join()
    for thread in consumer_threads:
        thread.join()
    
    print("Workflow completed!")

if __name__ == "__main__":
    run_kafka_workflow()
