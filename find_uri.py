from zenml.client import Client
print(Client().active_stack.experiment_tracker.get_tracking_uri())