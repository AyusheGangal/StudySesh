```JSON
{
  "mcpServers": {
    "job-posting-mcp": {
      "command": "python3",
      "args": [
        "/Users/ayushe/repos/mcp-learnings/job-posting-mcp/client_server.py"
      ],
      "env": {}
    },
    "devtools-ai-mock": {
      "command": "devtools-ai-mock-mcp-ayushe",
      "args": [],
      "env": {}
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/ayushe/repos"
      ]
    }
  }
}
```

can i package this as wheel and only use only "devtools-ai-mock-mcp-ayushe" in the config file instead of the entire path
"devtools-ai-mock": {
      "command": "devtools-ai-mock-mcp-ayushe",
      "args": [],
      "env": {}
    }


```JSON
{
  "mcpServers": {
    "job-posting-mcp": {
      "command": "python3",
      "args": [
        "/Users/ayushe/repos/mcp-learnings/job-posting-mcp/client_server.py"
      ],
      "env": {}
    },
    "devtools-ai-mock": {
      "command": "devtools-ai-mock-mcp-ayushe",
      "args": [],
      "env": {}
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/ayushe/repos"
      ]
    },
    "devtools-ai-mock-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "/Users/ayushe/repos/devtools-mock-mcp/devtools_ai_mock_mcp/fastmcp_server.py"
      ]
    }
  }
}
```

```JSON
{
  "mcpServers": {
    "devtools-ai-mock": {
      "command": "python3",
      "args": [
        "/Users/ayushe/repos/devtools-mock-mcp/devtools_ai_mock_mcp/proxy_server.py"
      ]
    }
  }
}

```

```JSON
{
  "mcpServers": {
    "job-posting-mcp": {
      "command": "python3",
      "args": [
        "/Users/ayushe/repos/mcp-learnings/job-posting-mcp/client_server.py"
      ],
      "env": {}
    },
    "devtools-ai-mock": {
      "command": "/Users/ayushe/opt/anaconda3/envs/agangal/bin/devtools-ai-mock-mcp-proxy"
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/ayushe/repos"
      ]
    }
  }
}
```