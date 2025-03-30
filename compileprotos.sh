source .venv/bin/activate
rm -rf ./contracts/*
python -m grpc_tools.protoc   -I=proto   --python_out=contracts   --grpc_python_out=contracts  proto/*.proto