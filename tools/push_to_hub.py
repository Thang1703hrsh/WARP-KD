from huggingface_hub import HfApi, HfFolder, Repository, create_repo
import os
import argparse

def push_to_hub(
    repo_name=None, 
    token=None, 
    commit_message="Upload model", 
    local_dir="./temp_model", 
    private=False
):
    try:
        if not repo_name:
            raise ValueError("must specify a repo name to push to hub")
        
        if not os.path.exists(local_dir):
            raise ValueError(f"local_dir {local_dir} does not exist")
        
        print(f"Pushing model to the hub at {repo_name}...")
        api = HfApi()
        create_repo(repo_name, token=token, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name, 
            token=token, 
            commit_message=commit_message
        )

        print(f"Model has been pushed to the hub at: {repo_name}")
        return True
        
    except Exception as e:
        print(f"Error pushing to hub: {str(e)}")
        return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", type=str, default="")
    parser.add_argument("--local-dir", type=str, default="")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--commit-message", type=str, default="")
    
    args = parser.parse_args()
    
    # manual args
    args.token = "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"
    args.commit_message = "Upload results to hf repo"
    
    push_to_hub(
        repo_name=args.repo_name,
        token=args.token,
        local_dir=args.local_dir,
        commit_message=args.commit_message,
        private=False
    )