import requests
import time

class PaliGemmaVerifier:
    def __init__(self, 
                 port:int=5000):
        self.server_url = f"http://localhost:{port}"

    def verify(self, image_path, prompt):
        return self.verify_batch([image_path], prompt)

    def verify_batch(self, image_paths, prompt: str):
        """Verify images as a batch in one forward pass."""

        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")

        # print(f"Process {len(image_paths)} images in one batch.")
        start_time = time.time()

        items = [{"image_path": img, "prompt": prompt} for img in image_paths]
        response = requests.post(
            f"{self.server_url}/paligemma/verify_batch",
            json={"items": items}
        )
        response.raise_for_status()
        # gets back raw results, usually: <prompt>\n<answer>
        raw_results = response.json()["results"] 
        
        elapsed_time = time.time() - start_time
        # print(f"Batch processing completed in {elapsed_time:.2f}s")
        # print(f"Average time per image: {elapsed_time/len(image_paths):.2f}s")
        
        # convert results to list format: [[<prompt>], [<answer>]]
        results = [r.split('\n') for r in raw_results]

        # Fetch out answers with wrong structure
        # Current behavior: PaliGemma failures return unclear!
        return [res[-1] if isinstance(res, list) else "error" for res in results]
    
    def corssref_results(self, verdicts, image_paths):
        
        confirmed = []
        rejected = []
        unclear = []

        # Ensure that verdicts and image paths are lists
        if not isinstance(verdicts, list): list(verdicts)
        if not isinstance(image_paths, list): list(image_paths)

        for r, image in zip(verdicts, image_paths):
            r = r.lower()
            if "yes" in r and "no" not in r:
                confirmed.append(image)
            elif "no" in r and "yes" not in r:
                rejected.append(image)
            else:
                unclear.append(image)

        return confirmed, rejected, unclear