PROTOC=protoc
PROTO_DIR=proto
OUT_DIR=.

proto:
	python -m grpc_tools.protoc \
  -I. \
  --python_out=$(OUT_DIR) \
  --grpc_python_out=$(OUT_DIR) \
  $(PROTO_DIR)/*.proto
.PHONY: proto

run:
	python main_ray.py -d 
.PHONY: run

gh:
	ghz \
		--insecure \
		--proto ./proto/translation.proto \
		--call translation.TranslationService.TranslateText \
		--data '{"source_text":"How are you? I am tired", "target_lang_code": "vi"}' \
		-c 10 \
		-n 1000 \
		--timeout 1s \
		localhost:50051
.PHONY: gh


ray:
	ray start --head

ray-serve:
	serve deploy serve_config.yaml

ray-stop:
	ray stop