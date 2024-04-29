from flask import Flask, request, jsonify
from dotenv import load_dotenv
from mongoengine import connect
import os
import smart_open
import redis
import google.generativeai as genai
from datetime import datetime
from mongoengine import (
    Document,
    DateTimeField,
    FloatField,
    ListField,
    StringField,
)


class RecordingEmbedding(Document):
    room_id = StringField(required=True)
    text_content = StringField(required=True)
    embeddings = ListField(FloatField(), required=True)
    created_at = DateTimeField(default=datetime.now().replace(microsecond=0))

    meta = {
        "collection": "recording_embedding",
        "indexes": [
            {"fields": ["room_id"]},
        ],
    }


app = Flask(__name__)


def extract_text_embedding(chunk: str) -> list:
    """
    Generate text embeddings for a text chunk using a pre-trained model.

    Args:
        chunk (str): The text chunk for which embeddings are to be generated.

    Returns:
        list: A list of embedding vectors representing the text chunk.

    Raises:
        Exception: If an error occurs during the embedding generation process.

    """
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="semantic_similarity",
        )
        return result["embedding"]
    except Exception as error:
        print(f"Error: {error}")
        raise


@app.route("/", methods=["GET"])
def index():
    return (
        jsonify({"message": "Webhook received successfully", "success": True}),
        200,
    )

@app.route("/api/recording-webhook", methods=["POST"])
def recording_webhook_listener():
    try:
        if request.is_json:
            webhook_data = request.get_json()

            if webhook_data.get("type") == "transcription.success":
                transcription_data = webhook_data.get("data")

                room_id = transcription_data.get("room_id")
                transcript_txt_presigned_url = transcription_data.get(
                    "transcript_txt_presigned_url"
                )

                load_dotenv()

                connect(
                    db=os.getenv("MONGO_DB"),
                    host=os.getenv("MONGO_URI"),
                    username=os.getenv("MONGO_USERNAME"),
                    password=os.getenv("MONGO_PASSWORD"),
                    alias="default",
                )

                text_content = None

                with smart_open.open(
                    transcript_txt_presigned_url, "rb"
                ) as transcript_file:
                    text_content = transcript_file.read().decode("utf-8")

                embedding_docs = []

                num_chunks = len(text_content)
                counter = 0

                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

                for i in range(0, num_chunks, 1000):
                    chunk = text_content[i : i + 1000]
                    embedding = extract_text_embedding(chunk)
                    counter += 1
                    embedding_doc = RecordingEmbedding(
                        room_id=room_id,
                        text_content=chunk,
                        embeddings=embedding,
                    )
                    embedding_docs.append(embedding_doc)

                RecordingEmbedding.objects.insert(embedding_docs, load_bulk=False)

                recording_number_of_embeddings_key = (
                    f"room_id_{room_id}_number_of_recording_embeddings"
                )

                redis_url = os.getenv("REDIS_URL")

                redis_client = redis.from_url(redis_url)

                with redis_client.pipeline() as pipe:
                    try:
                        existing_value = pipe.get(recording_number_of_embeddings_key)
                        if existing_value:
                            pipe.incrby(recording_number_of_embeddings_key, counter)
                        else:
                            pipe.set(recording_number_of_embeddings_key, counter)

                        pipe.execute()
                    except redis.exceptions.RedisError as error:
                        print(f"Error updating recording embeddings count: {error}")
                    else:
                        print(
                            f"Recording embeddings count updated for room_id: {room_id}"
                        )

            return (
                jsonify({"message": "Webhook received successfully", "success": True}),
                200,
            )

        return (
            jsonify({"error": "Invalid JSON data in request", "success": False}),
            400,
        )
    except Exception as error:
        return (
            jsonify({"error": str(error), "success": False}),
            500,
        )


if __name__ == "__main__":
    app.run()
