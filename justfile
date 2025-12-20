fmt:
    uv run autoflake --remove-all-unused-imports --remove-unused-variables --recursive --in-place .
    uv run isort -l 150 .
    uv run black -l 150 .
