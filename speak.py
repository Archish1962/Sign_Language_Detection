import paho.mqtt.client as mqtt
import os

# MQTT configurations
broker_address = ""   # your broker IP
topic = "sign_language/sentence"                # topic name

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker successfully!")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    message = msg.payload.decode('utf-8').strip()
    print(f"Received: {message}")

    if '.' in message:
        # Take only till the full stop
        sentence = message.split('.')[0]
        print(f"Final Sentence: {sentence}")

        # Save to text file
        with open('received_text.txt', 'w', encoding='utf-8') as file:
            file.write(sentence)

        # Ask user for language
        print("\nAvailable languages in eSpeak are limited, like:")
        print("en (English), hi (Hindi), kn (Kannada), ta (Tamil), te (Telugu), etc.")
        language_code = input("Enter language code: ").strip()

        try:
            # Speak using espeak
            speak_command = f'espeak -v {language_code} "{sentence}"'
            print(f"Speaking out: {sentence} in {language_code}...")
            os.system(speak_command)
        except Exception as e:
            print(f"Error during speaking: {e}")

# MQTT Client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, 1883, 60)

# Start listening
client.loop_forever()