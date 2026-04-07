"""
ComfyUI Integration Example: Control Allama from ComfyUI workflows
Enables: Image generation → Vision analysis → Feedback loop

Example:
  1. ComfyUI generates image on GPU0
  2. Allama analyzes image on GPU1 (Vision 7B)
  3. Send analysis back to ComfyUI
  4. Iterate for refinement
"""

import json
import requests
from typing import Dict, List, Optional
from pathlib import Path


class AllamaComfyUIBridge:
    """Bridge between ComfyUI and Allama for content creation pipelines."""

    def __init__(
        self,
        allama_host: str = "http://127.0.0.1",
        allama_port: int = 9000,
        profile: str = "research"
    ):
        """
        Initialize bridge.

        Args:
            allama_host: Allama server host
            allama_port: Allama server port
            profile: Which profile to use (research, ocr, etc)
        """
        self.base_url = f"{allama_host}:{allama_port}"
        self.profile = profile
        self.session = requests.Session()

    def analyze_generated_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image. What would make it better?"
    ) -> str:
        """
        Send generated image to Allama for analysis.

        Args:
            image_path: Path to generated image
            prompt: Analysis prompt

        Returns:
            Analysis text from Allama
        """
        # Read image and convert to base64
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Prepare request
        payload = {
            "model": self.profile,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "stream": False,
            "temperature": 0.7
        }

        # Call Allama
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def batch_analyze_images(
        self,
        image_dir: str,
        prompt: str,
        output_file: str = "analysis.json"
    ) -> Dict:
        """
        Analyze multiple generated images.

        Args:
            image_dir: Directory containing images
            prompt: Analysis prompt
            output_file: Save results here

        Returns:
            Dict mapping image filenames to analysis
        """
        image_dir = Path(image_dir)
        results = {}

        for image_path in sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")):
            print(f"Analyzing {image_path.name}...", end=" ", flush=True)

            try:
                analysis = self.analyze_generated_image(str(image_path), prompt)
                results[image_path.name] = {
                    "path": str(image_path),
                    "analysis": analysis,
                    "status": "success"
                }
                print("✓")
            except Exception as e:
                results[image_path.name] = {
                    "path": str(image_path),
                    "error": str(e),
                    "status": "error"
                }
                print(f"✗ ({e})")

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")
        return results

    def extract_text_from_image(
        self,
        image_path: str,
        preserve_layout: bool = True
    ) -> str:
        """
        Extract text from image (OCR-like).

        Args:
            image_path: Path to image
            preserve_layout: Try to preserve document layout

        Returns:
            Extracted text
        """
        profile_to_use = "ocr" if preserve_layout else self.profile

        prompt = "Extract all text from this image. Preserve layout and formatting."

        # Call Allama with OCR profile
        analysis = self.analyze_generated_image(image_path, prompt)
        return analysis

    def get_image_suggestions(
        self,
        image_path: str,
        style: str = "alternative versions",
        num_suggestions: int = 3
    ) -> List[str]:
        """
        Get suggestions for image variations.

        Args:
            image_path: Path to current image
            style: Type of suggestions (alternative versions, different angles, etc)
            num_suggestions: Number of suggestions to generate

        Returns:
            List of prompts for ComfyUI generation
        """
        prompt = f"""Analyze this image and suggest {num_suggestions} creative variations in {style}.
Format as a numbered list. Be specific about:
- Composition changes
- Lighting adjustments
- Color palette modifications
- Element positioning

Focus on improvements that would enhance the overall quality."""

        analysis = self.analyze_generated_image(image_path, prompt)

        # Parse suggestions
        suggestions = [line.strip() for line in analysis.split("\n") if line.strip()]
        return suggestions[:num_suggestions]


class ComfyUIWorkflow:
    """Helper class to integrate with ComfyUI API."""

    def __init__(self, comfyui_host: str = "http://127.0.0.1", comfyui_port: int = 8188):
        self.base_url = f"{comfyui_host}:{comfyui_port}"

    def queue_prompt(self, workflow: Dict) -> str:
        """Queue a prompt in ComfyUI and return execution ID."""
        response = requests.post(f"{self.base_url}/prompt", json=workflow)
        response.raise_for_status()
        return response.json()["prompt_id"]

    def get_latest_image(self, output_dir: str = "/path/to/output") -> Optional[str]:
        """Get path to most recently generated image."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return None

        images = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
        if not images:
            return None

        return str(max(images, key=lambda p: p.stat().st_mtime))


# ============================================================================
# EXAMPLE PIPELINE: Auto-Refine Generated Images
# ============================================================================

def auto_refine_pipeline(
    comfyui_workflow_path: str,
    comfyui_output_dir: str,
    target_quality: str = "professional photography",
    max_iterations: int = 3
):
    """
    Auto-refine pipeline:
    1. Generate image with ComfyUI
    2. Analyze with Allama Vision
    3. Get improvement suggestions
    4. Regenerate with suggestions
    5. Repeat until satisfied
    """
    bridge = AllamaComfyUIBridge(profile="research")
    comfyui = ComfyUIWorkflow()

    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")

        # Step 1: Generate image
        print("1️⃣  Generating image with ComfyUI...")
        with open(comfyui_workflow_path) as f:
            workflow = json.load(f)
        prompt_id = comfyui.queue_prompt(workflow)
        print(f"   Queue ID: {prompt_id}")

        # Step 2: Get generated image
        print("2️⃣  Waiting for image output...")
        import time
        time.sleep(5)  # Wait for generation
        image_path = comfyui.get_latest_image(comfyui_output_dir)
        if not image_path:
            print("   ❌ No image generated!")
            break
        print(f"   ✓ Generated: {image_path}")

        # Step 3: Analyze with Allama
        print("3️⃣  Analyzing with Allama Vision...")
        analysis = bridge.analyze_generated_image(
            image_path,
            f"Evaluate this image against '{target_quality}'. "
            "What could be improved? Be specific and constructive."
        )
        print(f"   Analysis:\n{analysis}")

        # Step 4: Get improvement suggestions
        print("4️⃣  Getting improvement suggestions...")
        suggestions = bridge.get_image_suggestions(
            image_path,
            style="improvements for better composition"
        )
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")

        # Step 5: Check if quality is acceptable
        quality_check = bridge.analyze_generated_image(
            image_path,
            f"Rate this image's quality for '{target_quality}' on 1-10. "
            "Reply with just the number."
        )

        try:
            rating = int(quality_check.strip().split()[0])
            print(f"5️⃣  Quality rating: {rating}/10")

            if rating >= 7 or iteration == max_iterations - 1:
                print(f"   ✓ Satisfied! Moving forward with this image.")
                break
        except (ValueError, IndexError):
            print(f"5️⃣  Could not parse quality rating")

        # Prepare for next iteration
        print(f"   → Will regenerate with suggestions in next iteration...")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Final image: {image_path}")
    print(f"{'='*60}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI + Allama Integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze = subparsers.add_parser("analyze", help="Analyze a generated image")
    analyze.add_argument("image_path", help="Path to image")
    analyze.add_argument("--prompt", default="Analyze this image. What's good? What could improve?")
    analyze.add_argument("--profile", default="research")

    # Batch analyze
    batch = subparsers.add_parser("batch", help="Batch analyze images")
    batch.add_argument("image_dir", help="Directory with images")
    batch.add_argument("--prompt", default="Analyze this image quality and suggest improvements.")
    batch.add_argument("--output", default="analysis.json")

    # Extract text
    ocr = subparsers.add_parser("ocr", help="Extract text from image")
    ocr.add_argument("image_path", help="Path to image")

    # Get suggestions
    suggest = subparsers.add_parser("suggest", help="Get image variation suggestions")
    suggest.add_argument("image_path", help="Path to image")
    suggest.add_argument("--style", default="alternative compositions")
    suggest.add_argument("--count", type=int, default=3)

    # Auto-refine
    refine = subparsers.add_parser("refine", help="Auto-refine images iteratively")
    refine.add_argument("workflow", help="ComfyUI workflow JSON path")
    refine.add_argument("output_dir", help="ComfyUI output directory")
    refine.add_argument("--quality", default="professional photography")
    refine.add_argument("--iterations", type=int, default=3)

    args = parser.parse_args()

    if args.command == "analyze":
        bridge = AllamaComfyUIBridge(profile=args.profile)
        result = bridge.analyze_generated_image(args.image_path, args.prompt)
        print(result)

    elif args.command == "batch":
        bridge = AllamaComfyUIBridge()
        results = bridge.batch_analyze_images(args.image_dir, args.prompt, args.output)

    elif args.command == "ocr":
        bridge = AllamaComfyUIBridge(profile="ocr")
        result = bridge.extract_text_from_image(args.image_path)
        print(result)

    elif args.command == "suggest":
        bridge = AllamaComfyUIBridge()
        suggestions = bridge.get_image_suggestions(args.image_path, args.style, args.count)
        for i, s in enumerate(suggestions, 1):
            print(f"{i}. {s}")

    elif args.command == "refine":
        auto_refine_pipeline(args.workflow, args.output_dir, args.quality, args.iterations)

    else:
        parser.print_help()
