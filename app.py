import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import gradio as gr
import random

# Load the model architecture and weights
model = TFDistilBertForSequenceClassification.from_pretrained("spam_detection_model/")

# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# Define the process_input function
def process_input(text):
    # Preprocess the sample text using the tokenizer
    encodings = tokenizer(text, truncation=True, padding=True, return_tensors="tf")

    # Perform inference
    logits = model(encodings.input_ids).logits

    # Convert logits to probabilities using softmax
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Get the predicted class
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]

    # Map the predicted class to label
    label_mapping = {
        0: '<b><div style="font-size:16px; text-align:center;">No need to worry, Not a spam message.</div></b>',
        1: '<b><div style="font-size:16px; color:#ff3b5c; text-align:center;">Warning⚠️: This message has been identified as spam.</div></b>',
    }
    predicted_label = label_mapping[predicted_class]

    return [
        {
            "Spam": float(probabilities.numpy()[0][1]),
            "Not a Spam": float(probabilities.numpy()[0][0]),
        },
        predicted_label,
    ]


# Define the Gradio interface
title = "Spam Detector⚠️"
examples = [
    "Dear Customer, Your account has been compromised. Click the link below to verify your account details immediately or risk suspension.",
    "You've been selected as the lucky winner of our international sweepstakes! To claim your prize, reply with your full name, address, and bank details.",
    "Congratulations! You've won a free iPhone X. Click the link to claim your prize.",
    "URGENT: Your bank account has been compromised. Click here to reset your password.",
    "Get rich quick! Invest in our exclusive program and earn thousands overnight.",
    "Your prescription refill is ready for pickup at your local pharmacy. Visit us at your convenience",
    "Reminder: Your monthly utility bill is due on August 20th. Please make the payment.",
    "You've been selected as the lucky winner of a million-dollar lottery. Reply to claim.",
    "Limited time offer: Double your money with our amazing investment opportunity.",
    "Hi, just checking in to see how you're doing. Let's catch up soon.",
    "Reminder: Your dentist appointment is scheduled for tomorrow at 2 PM.",
    "Invitation: Join us for a webinar on digital marketing strategies. Register now!",
    "Your application for the scholarship has been reviewed. We're pleased to inform you that you've been selected.",
    'Hi there! Just wanted to check in and see how you\'re doing.',
    'Reminder: Your friend\'s birthday is coming up. Don\'t forget to send them a message.',
    'Thank you for your purchase. Your order has been successfully processed.',
    'Your monthly newsletter is here! Stay updated with the latest news and updates.',
    'Invitation: Join us for a community clean-up event this weekend. Let\'s make a difference together.',
    'Reminder: Your scheduled appointment is tomorrow. We look forward to seeing you.',
    'Good news! You\'ve earned a reward for your loyalty. Check your account for details.',
    'Your recent transaction has been approved. Please keep this email for your records.',
    'Exciting announcement: Our new store location is now open. Visit us and receive a special discount.',
    'Welcome to our online community! Here\'s how to get started and connect with others.',
    'Your request has been received and is being processed. We\'ll update you with the status soon.',
    'Upcoming event: Join us for a free cooking class this Saturday. Learn new recipes and techniques.',
    'Reminder: Don\'t forget to vote in the upcoming election. Your voice matters.',
    'Join our book club and dive into a world of fascinating stories. Here\'s how to join.',
    'A few things I should specify: By "tablet" I\'m referring to tablet computers running iOS or Android, and by "PC" I\'m referring to computers running Mac OS, Windows, our Linux. I\'m particularly interested in usage, rather than just ownership. Also, I\'m looking for a dataset that\'s broken down by age bracket, as I\'m particularly interested in the 65+ demographic. Thanks!',
    'I wanted it to implement simple search algorithms on it.',
    'Are you tired of constantly feeling left out of the loop? Do you find yourself overwhelmed and unfulfilled with the mundane aspects of life? Well, fear not my friends because [insert social network name here] is here to save the day! Our platform offers an array of mindless scrolling opportunities that will surely cure your boredom. From uninteresting memes to recycled TikTok videos, we have it all. But that\'s not all folks! We also offer the exclusive opportunity to follow and engage with random strangers, because why not connect with people you most likely will never meet in person? We guarantee that your time spent on our app will be',
    'Guaranteed weight loss in 3 days!!! Try our magical diet pills and lose weight fast and easy.',
]



# Create Gradio components
input_text = gr.Textbox(
    lines=3, label="Enter the SMS/Message/Email you received", autofocus=True
)
output_text = gr.HTML("", label="Output")
probabilities_text = gr.Label("", label="Probabilities")

random.shuffle(examples)

# Initialize the Gradio interface
model_gui = gr.Interface(
    fn=process_input,
    inputs=input_text,
    outputs=[probabilities_text, output_text],
    title=title,
    examples=examples,
    interpretation="default",
    theme="shivi/calm_seafoam",
    css="""*{font-family:'IBM Plex Mono';}""",
    examples_per_page=15,
)

# Launch the Gradio interface
print("add '/?__theme=dark' to URL for rendering in dark theme.")
model_gui.launch()
