# Use Python 3.13
FROM python:3.13-bookworm

ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update
RUN apt-get install cmake curl build-essential -y
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN poetry config virtualenvs.create false

# Set the working directory
WORKDIR /app

# Install Playwright dependencies and browsers
# Install this before doing the full requirements install because it's slow and will rarely change
# the poetry.toml will change more often, so re-builds will be faster if this is on a lower layer
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN pip install playwright
RUN playwright install --with-deps

# Copy the requirements file and install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry lock --no-update
RUN poetry install --no-root

# yeah install it again because it always asks for an update.
# It's still generally faster to do this "twice" as the first install is cached in a Docker layer
RUN playwright install

# Copy the app files (if any)
COPY . .

EXPOSE 5000
EXPOSE 5678

# Default command: start MCP server
CMD ["poetry", "run", "python", "-m", "agent.mcp_server"]
