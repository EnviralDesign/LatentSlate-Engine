"""LTX 2.5 enhancer prompt and image token inputs.

Adapted from ComfyUI 1a14b82e nodes_textgen.py and gemma4.py.
Image resizing preserves torchvision's uint8 rounding around Torch interpolation.
"""

import math
import re

import torch
from torch.nn import functional as F


def decode_enhancement(tokenizer, token_ids, original_prompt: str) -> str:
    """Remove Gemma channel markers and reasoning, retaining the empty fallback."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    text = re.sub(
        r"<\|channel>thought\n(.*?)<channel\|>",
        r"<think>\n\1</think>",
        text,
        flags=re.DOTALL,
    )
    text = text.replace("<|channel>thought\n", "<think>\n")
    text = re.sub(r"<\|channel>\w*\n?|<channel\|>|<\|turn>\w*\n?|<turn\|>", "", text)
    text = text.replace("<eos>", "").strip()
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    return text or original_prompt


LTX24_T2V_SYSTEM_PROMPT = 'You are given a user\'s short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user\'s ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.\n\nMatch this captioning style precisely:\n\n1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".\n\n2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").\n\n3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.\n\n4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):\n   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)\n   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.\n   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).\n   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".\n\n5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.\n\n6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.\n\n7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).\n\nIf the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.\n\nAESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.\n'

LTX24_I2V_SYSTEM_PROMPT = 'You are given a REFERENCE IMAGE (the exact first frame of the video) and a user\'s short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user\'s ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.\n\nFIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly — same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user\'s requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take — no hard cuts.\n\nMatch this captioning style precisely:\n\n1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".\n\n2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").\n\n3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.\n\n4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):\n   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the reference image is framed at the start.\n   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.\n   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the reference image\'s viewpoint at the opening.\n   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".\n\n5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.\n\n6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user\'s requested motion/action central and in motion throughout.\n\n7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).\n\nIf the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.\n\nAESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.\n'


def tokenize_enhancement(tokenizer, prompt: str, image: torch.Tensor | None = None):
    """Create the non-thinking E2B input used by the curated workflows."""
    system = (
        LTX24_T2V_SYSTEM_PROMPT if image is None else LTX24_I2V_SYSTEM_PROMPT
    ).strip()
    user = (
        f"user prompt: {prompt}"
        if image is None
        else f"User Raw Input Prompt: {prompt}."
    )
    media = "" if image is None else "<|image><|image|><image|>\n\n"
    text = (
        f"<|turn>system\n{system}<turn|>\n"
        f"<|turn>user\n{media}{user}<turn|>\n"
        "<|turn>model\n<|channel>final\n"
    )
    ids = [2] + tokenizer.encode(text, add_special_tokens=False).ids
    tokens = [(token, 1.0) for token in ids]
    if image is not None:
        height, width = image.shape[1:3]
        factor = math.sqrt(280 * 9 * 16**2 / (height * width))
        target_h = math.floor(factor * height / 48) * 48
        target_w = math.floor(factor * width / 48) * 48
        if target_h == 0:
            target_h, target_w = 48, min(math.floor(width / height) * 48, 280 * 48)
        elif target_w == 0:
            target_h, target_w = min(math.floor(height / width) * 48, 280 * 48), 48
        pixels = (image[:1].movedim(-1, 1).clamp(0, 1) * 255).to(torch.uint8)
        if (height, width) != (target_h, target_w):
            pixels = F.interpolate(
                pixels.float(),
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pixels = pixels.clamp(0, 255).round().to(torch.uint8)
        pixels = (pixels.float() * (1.0 / 255.0)).movedim(1, -1)[:, :, :, :3]
        index = ids.index(258880)
        tokens[index] = ({"type": "image", "data": pixels, "max_soft_tokens": 280}, 1.0)
    return tokens
