# 도움말 표시
default:
    @just --list

default_model:="claude-sonnet-4-5-20250514"
api_url:="https://api.anthropic.com/v1/messages"

# ============================================
# Chapter 2
# ============================================

test-no-cache:
	@echo "=== 캐싱 없는 기본 요청 ==="
	@curl -s {{api_url}} \
	  -H "content-type: application/json" \
	  -H "x-api-key: $$ANTHROPIC_API_KEY" \
	  -H "anthropic-version: 2023-06-01" \
	  -d @payloads/no_cache.json \
	| jq '.usage'

# 클린업
clean:
    @rm -f sample-content.txt
    @echo "Cleaned up temporary files."
