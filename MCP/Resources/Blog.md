Very important, well documented resouce:
https://simplescraper.io/blog/how-to-mcp

mcp job and file server
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

mcp job-posting
```JSON
{
  "mcpServers": {
    "job-posting-mcp": {
      "command": "python3",
      "args": [
        "/Users/ayushe/repos/mcp-learnings/job-posting-mcp/client_server.py"
      ],
      "env": {}
    }
  }
}
```