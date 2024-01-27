import base64
import openai
import os

from PIL import Image
from typing import Literal


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OFFICIAL_OPENAI_STRICT_PROMPT_HINT = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"


def gen_image(
    *,
    model: Literal["dall-e-3"],
    prompt: str,
    quality: Literal["standard", "hd"],
    dimensions: Literal["1024x1024", "1024x1792", "1792x1024"],
    style: Literal["vivid", "natural"],
    strict_prompt_adherence: bool,
    output_image_path: str,
) -> None:
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
    )

    if strict_prompt_adherence:
        prompt = f"{OFFICIAL_OPENAI_STRICT_PROMPT_HINT} {prompt}"

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=dimensions,
        quality=quality,
        n=1,
        response_format="b64_json",
        style=style,
    )

    data = response.data[0]

    print(f"Prompt: {prompt}")
    print(f"Revised prompt: {data.revised_prompt}")

    b64 = data.b64_json
    assert b64 is not None
    with open(output_image_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64))

    image = Image.open(output_image_path)
    image.show()


def main():
    prompt = "art of a dragon in a cavern of gold"
    gen_image(
        model="dall-e-3",
        prompt=prompt,
        quality="hd",
        dimensions="1024x1024",
        style="natural",
        strict_prompt_adherence=True,
        output_image_path="output.png",
    )


if __name__ == "__main__":
    main()
