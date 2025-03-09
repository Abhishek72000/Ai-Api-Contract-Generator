import os
import git
import json
import openai
import hmac
import hashlib
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from typing import Dict, List, Optional
from markdown import markdown as md_parser
import requests
import time
from pyngrok import ngrok
from flask_cors import CORS

# Load environment variables
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_PATH = os.getenv("REPO_PATH")
DOCS_PATH = os.getenv("DOCS_PATH")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
HF_API_KEY = os.getenv("HF_API_KEY")


# Validate required environment variables
if not all([HF_API_KEY, REPO_PATH, DOCS_PATH, WEBHOOK_SECRET]):
    raise ValueError("Missing required environment variables: OPENAI_API_KEY, REPO_PATH, DOCS_PATH, or WEBHOOK_SECRET")


app = Flask(__name__)
CORS(app)

class APIDocGenerator:
    def __init__(self, repo_path: str, docs_path: str):
        self.repo_path = repo_path
        self.docs_path = docs_path
        try:
            self.repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            raise ValueError(f"Invalid Git repository path: {repo_path}")

    def get_latest_changes(self, commit_hash: str) -> List[dict]:
        """Get files changed in the specified commit, handling all change types."""
        try:
            commit = self.repo.commit(commit_hash)
            print("commit",commit)
            print("commithash",commit_hash)
            diff_index = commit.diff(f"{commit_hash}~1")
            changed_files = []

            for diff_item in diff_index:
                change_type = diff_item.change_type  # 'A' (added), 'M' (modified), 'D' (deleted), etc.
                file_info = {
                    "path": diff_item.b_path or diff_item.a_path,  # Use b_path for current state
                    "change_type": change_type,
                    "old_path": diff_item.a_path if change_type != "A" else None,
                    "new_path": diff_item.b_path if change_type != "D" else None
                }
                if file_info["path"].endswith(('.cs', '.py')):
                    changed_files.append(file_info)
            return changed_files
        except git.exc.GitCommandError as e:
            print(f"Error fetching diff for commit {commit_hash}: {e}")
            return []
        

    def identify_api_changes(self, changed_files: List[dict]) -> List[dict]:
        api_changes = []
        for file_info in changed_files:
            file_path = file_info["path"]
            change_type = file_info["change_type"]
            if "Controller" in file_path and file_path.endswith(".cs"):
                controller_name = os.path.basename(file_path).replace("Controller.cs", "")
                if change_type == "D":
                    api_changes.append({"controller": controller_name, "type": "deleted"})
                    continue
                elif change_type == "A":
                    try:
                        new_content = self.repo.git.show(f"HEAD:{file_path}")
                        print(f"New content (A): {new_content[:500]}...")
                    except git.exc.GitCommandError as e:
                        print(f"Git error fetching new content for {file_path}: {e}")
                        new_content = ""
                    new_methods = self._extract_api_methods(new_content) if new_content else {}
                    for method_name, method_info in new_methods.items():
                        api_changes.append({
                            "controller": controller_name,
                            "method": method_name,
                            "type": "added",
                            "content": method_info["content"],
                            "http_method": method_info["http_method"],
                            "route": method_info["route"]
                        })
                elif change_type == "M":
                    try:
                        new_content = self.repo.git.show(f"HEAD:{file_path}")
                        print(f"New content (M): {new_content[:500]}...")
                    except git.exc.GitCommandError as e:
                        print(f"Git error fetching new content for {file_path}: {e}")
                        new_content = ""
                    try:
                        old_content = self.repo.git.show(f"HEAD~1:{file_path}")
                        print(f"Old content (M): {old_content[:500]}...")
                    except git.exc.GitCommandError as e:
                        print(f"Git error fetching old content for {file_path}: {e}")
                        old_content = ""
                    old_methods = self._extract_api_methods(old_content) if old_content else {}
                    new_methods = self._extract_api_methods(new_content) if new_content else {}
                    print(f"Old methods: {list(old_methods.keys())}")
                    print(f"New methods: {list(new_methods.keys())}")
                    
                    for method_name, method_info in new_methods.items():
                        old_method_info = old_methods.get(method_name)
                        if not old_method_info:
                            print(f"Added method detected: {method_name}")
                            api_changes.append({
                                "controller": controller_name,
                                "method": method_name,
                                "type": "added",
                                "content": method_info["content"],
                                "http_method": method_info["http_method"],
                                "route": method_info["route"]
                            })
                        elif old_method_info["content"] != method_info["content"]:
                            print(f"Modified method detected: {method_name}")
                            api_changes.append({
                                "controller": controller_name,
                                "method": method_name,
                                "type": "modified",
                                "content": method_info["content"],
                                "old_content": old_method_info["content"],
                                "http_method": method_info["http_method"],
                                "route": method_info["route"]
                            })
                    for method_name in old_methods:
                        if method_name not in new_methods:
                            print(f"Deleted method detected: {method_name}")
                            api_changes.append({
                                "controller": controller_name,
                                "method": method_name,
                                "type": "deleted"
                            })
        print("API changes:", api_changes)
        return api_changes
    
    

    def _extract_api_methods(self, content: str) -> dict:
        methods = {}
        lines = content.split('\n')
        current_http_method = None
        current_route = None
        method_content = []
        brace_count = 0
        in_method_body = False
        method_name = None
        
        http_method_pattern = r'\[Http(Get|Post|Put|Delete|Patch)\(?.*?\)?\]'
        route_pattern = r'\[Route\("([^"]+)"\)\]'
        method_pattern = r'(?:public|private|protected|internal)?\s*(?:async\s+)?(?:Task<)?(?:IActionResult|\w+)\s+(\w+)\s*\('
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('//'):
                i += 1
                continue
            
            http_match = re.search(http_method_pattern, line)
            if http_match:
                current_http_method = http_match.group(1)
                i += 1
                continue
            
            route_match = re.search(route_pattern, line)
            if route_match:
                current_route = route_match.group(1)
                i += 1
                continue
            
            method_match = re.search(method_pattern, line)
            if method_match and current_http_method:
                method_name = method_match.group(1)
                method_content = [line]  # Start with the signature line
                brace_count = 0
                in_method_body = False
                current_http_method_save = current_http_method
                current_http_method = None
                current_route_save = current_route
                current_route = None
                
                # Collect lines until brace_count returns to 0 after entering method body
                while i < len(lines):
                    current_line = lines[i].strip()
                    method_content.append(current_line)
                    brace_count += current_line.count('{') - current_line.count('}')
                    if brace_count > 0:
                        in_method_body = True  # We've entered the method body
                    if in_method_body and brace_count == 0:
                        break  # We've exited the method body
                    i += 1
                
                if brace_count == 0 and in_method_body:
                    methods[method_name] = {
                        "content": "\n".join(method_content),
                        "http_method": current_http_method_save,
                        "route": current_route_save or "Unknown"
                    }
                else:
                    print(f"Warning: Unbalanced braces for method {method_name}")
                method_name = None
            else:
                i += 1
        
        print(f"Extracted methods: {list(methods.keys())}")
        return methods

    def generate_api_documentation(self, api_change: dict) -> str:
        """Generate API documentation using Mistral 7B Instruct via Hugging Face, optimized for token usage."""
        if api_change["type"] == "deleted":
            if "method" in api_change:
                return f"## {api_change['method']}\n\n**Status**: This endpoint has been removed."
            else:
                return f"# {api_change['controller']} API Documentation\n\n**Status**: This controller has been removed."
        
        # Construct prompt
        prompt_parts = [
            f"Generate API documentation for this {api_change['http_method']} endpoint:",
            f"Controller: {api_change['controller']}",
            f"Method: {api_change['method']}",
            f"Route: {api_change['route']}",
            "Code:",
            "```csharp",
            api_change['content'].strip(),
            "```",
            "Provide (CRITICAL: Base all descriptions EXCLUSIVELY on the code above. Do NOT infer logic from the method name '{api_change['method']}'; for example, if the code uses modulo 5, describe modulo 5, not 13):",
            "- A detailed description of the logic based ONLY on the provided code.",
            "- Input parameters (name, type, description).",
            "- A line-by-line explanation of the logic in the method's body ONLY, based solely on the code.",
            "- Format in Markdown.",
            "START_DOCUMENTATION_HERE"  # Explicit marker
        ]
        
        if api_change["type"] == "modified" and "old_content" in api_change:
            prompt_parts.append(f"Previous implementation: {api_change['old_content'].strip()}")
            prompt_parts.append("Highlight key differences and their impact.")
        
        prompt = "\n".join(prompt_parts).strip()
        
        hf_api_key = os.getenv("HF_API_KEY")
        if not hf_api_key:
            print("Missing HF_API_KEY in environment variables.")
            return f"## {api_change['method']}\n\nError: Missing Hugging Face API key."
        
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        model_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.2,
                "max_new_tokens": 1500
            }
        }
        
        max_retries = 3
        delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(model_api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    generated_text = (result[0].get("generated_text", "")
                                    if isinstance(result, list)
                                    else result.get("generated_text", ""))
                    # Explicitly strip prompt by finding the marker
                    marker = "START_DOCUMENTATION_HERE"
                    if marker in generated_text:
                        doc_content = generated_text.split(marker, 1)[1].strip()
                    else:
                        # Fallback: strip everything before first Markdown header
                        lines = generated_text.split('\n')
                        start_idx = next((i for i, line in enumerate(lines) if line.startswith('##')), 0)
                        doc_content = '\n'.join(lines[start_idx:]).strip()
                    return doc_content
                elif response.status_code == 503:
                    print(f"HuggingFace API 503 error on attempt {attempt + 1}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"HuggingFace API error: {response.status_code} - {response.text}")
                    return f"## {api_change['method']}\n\nError: Documentation generation failed."
            except Exception as e:
                print(f"Exception during HuggingFace API call: {e}")
                return f"## {api_change['method']}\n\nError: Documentation generation failed."
        
        return f"## {api_change['method']}\n\nError: Documentation generation failed after multiple attempts."



    def update_documentation(self, api_change: dict, generated_doc: str) -> str:
        """Update or create documentation file with unique markers."""
        controller_doc_path = os.path.join(self.docs_path, f"{api_change['controller']}_API.md")
        os.makedirs(os.path.dirname(controller_doc_path), exist_ok=True)
        
        if "method" not in api_change:
            with open(controller_doc_path, 'w') as f:
                f.write(generated_doc)
        else:
            method_name = api_change['method']
            start_marker = f"<!-- START_METHOD: {method_name} -->"
            end_marker = f"<!-- END_METHOD: {method_name} -->"
            
            # Minimal cleaning, trusting generate_api_documentation
            clean_doc = generated_doc.strip()
            new_method_doc = f"{start_marker}\n{clean_doc}\n{end_marker}"
            
            if os.path.exists(controller_doc_path):
                with open(controller_doc_path, 'r') as f:
                    content = f.read()
                pattern = re.compile(f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
                if api_change["type"] == "deleted":
                    updated_content = re.sub(pattern, "", content).strip()
                else:
                    if pattern.search(content):
                        updated_content = re.sub(pattern, new_method_doc, content)
                    else:
                        updated_content = content.rstrip() + "\n\n" + new_method_doc
            else:
                header = f"# {api_change['controller']} API Documentation\n\n"
                updated_content = header + new_method_doc
            
            with open(controller_doc_path, 'w') as f:
                f.write(updated_content.strip())
        
        return controller_doc_path
def verify_webhook_signature(payload: bytes, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
    
@app.route('/webhook', methods=['POST', 'GET'])
def git_webhook():
        if request.method == 'GET':
            return "Webhook is active!", 200 


        """Handle Git webhook payload."""
        # if not verify_webhook_signature(request.data, WEBHOOK_SECRET):
        #     return jsonify({"status": "error", "message": "Invalid signature"}), 403

        payload = request.json
        if not payload or "after" not in payload:
            return jsonify({"status": "error", "message": "Invalid payload"}), 400

        commit_hash = payload["after"]

        # ------------------------------------------------------------
        repo = git.Repo(REPO_PATH)
        print(f"Repository at: {REPO_PATH}, HEAD: {repo.head.commit.hexsha}")
        if repo.is_dirty():
            print("Warning: Repository has uncommitted changes.")
        
        try:
            commit = repo.commit(commit_hash)
            parent = commit.parents[0] if commit.parents else None
            print(f"Commit: {commit.hexsha}, Parent: {parent.hexsha if parent else 'None'}")
        except git.exc.BadName:
            print(f"Error: Commit hash {commit_hash} not found.")
            return
        
        # Use raw Git diff for reliability
        diff_output = repo.git.diff("--name-status", parent.hexsha if parent else "4b825dc642cb6eb9a060e54bf8d69288fbee4904", commit.hexsha)
        changed_files = []
        for line in diff_output.splitlines():
            status, path = line.split(maxsplit=1)
            change_type = status
            old_path = path if change_type == "D" else None
            new_path = path if change_type in ["A", "M"] else None
            changed_files.append({
                "path": path,
                "change_type": change_type,
                "old_path": old_path,
                "new_path": new_path
            })
        
        print("changedfiles", changed_files)
        # ------------------------------------------------------------

        generator = APIDocGenerator(REPO_PATH, DOCS_PATH)

        # changed_files = generator.get_latest_changes(commit_hash)
        api_changes = generator.identify_api_changes(changed_files)

        results = []
        for api_change in api_changes:
            generated_doc = generator.generate_api_documentation(api_change)
            doc_path = generator.update_documentation(api_change, generated_doc)
            results.append({
                "controller": api_change["controller"],
                "method": api_change.get("method", "N/A"),
                "type": api_change["type"],
                "documentation_path": doc_path
            })

        response_data = {
        "status": "success",
        "changes_processed": len(api_changes),
        "results": results
        }

        print(json.dumps(response_data, indent=2)) 

        return jsonify(response_data), 200  # 


def test_locally(commit_hash: str):
    generator = APIDocGenerator(REPO_PATH, DOCS_PATH)
    
    repo = git.Repo(REPO_PATH)
    print(f"Repository at: {REPO_PATH}, HEAD: {repo.head.commit.hexsha}")
    if repo.is_dirty():
        print("Warning: Repository has uncommitted changes.")
    
    try:
        commit = repo.commit(commit_hash)
        parent = commit.parents[0] if commit.parents else None
        print(f"Commit: {commit.hexsha}, Parent: {parent.hexsha if parent else 'None'}")
    except git.exc.BadName:
        print(f"Error: Commit hash {commit_hash} not found.")
        return
    
    # Use raw Git diff for reliability
    diff_output = repo.git.diff("--name-status", parent.hexsha if parent else "4b825dc642cb6eb9a060e54bf8d69288fbee4904", commit.hexsha)
    changed_files = []
    for line in diff_output.splitlines():
        status, path = line.split(maxsplit=1)
        change_type = status
        old_path = path if change_type == "D" else None
        new_path = path if change_type in ["A", "M"] else None
        changed_files.append({
            "path": path,
            "change_type": change_type,
            "old_path": old_path,
            "new_path": new_path
        })
    
    print("changedfiles", changed_files)
    api_changes = generator.identify_api_changes(changed_files)
    print("apichanges", api_changes)
    results = []
    # Uncomment to enable documentation generation
    for api_change in api_changes:
        generated_doc = generator.generate_api_documentation(api_change)
        doc_path = generator.update_documentation(api_change, generated_doc)
        results.append({
            "controller": api_change["controller"],
            "method": api_change.get("method", "N/A"),
            "type": api_change["type"],
            "documentation_path": doc_path
        })
    print(json.dumps({
        "status": "success",
        "changes_processed": len(api_changes),
        "results": results
    }, indent=2))
if __name__ == "__main__":
    app.run(host="127.0.0.1",debug=True, port=8000)

    # test_locally("28d4d57")  # Use "HEAD" or a specific commit hash