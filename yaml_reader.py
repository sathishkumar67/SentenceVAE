import yaml

# Load the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Function to get a specific model configuration by name
def get_model_config(model_name):
    for model in config['models']:
        if model['name'] == model_name:
            return model
    return None

# Example usage:
model_name = 'SVAE-1024-H4'
model_config = get_model_config(model_name)

if model_config:
    print(f"Configuration for {model_name}:")
    print(model_config)
else:
    print(f"Model {model_name} not found in the configuration.")
