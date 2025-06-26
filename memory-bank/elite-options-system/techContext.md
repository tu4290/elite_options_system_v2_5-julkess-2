# Elite Options System v2.5 - Technical Context

## Core Technologies

### Programming Language & Runtime
- **Python 3.10+**: Primary development language
- **Async/Await**: Concurrent processing for data fetching and analytics
- **Type Hints**: Full type annotation for better code quality

### Data Processing Stack
- **pandas**: Primary data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scipy**: Statistical functions and mathematical operations
- **polars**: High-performance alternative for large datasets (future consideration)

### Visualization & UI
- **plotly**: Interactive charting and visualization
- **dash**: Web application framework for dashboard
- **dash-bootstrap-components**: UI components and styling
- **plotly-express**: Simplified plotting interface

### Data Validation & Contracts
- **pydantic**: Data validation and serialization
- **typing**: Type system extensions
- **dataclasses**: Structured data containers

### HTTP & API Integration
- **requests**: Synchronous HTTP client
- **aiohttp**: Asynchronous HTTP client
- **httpx**: Modern HTTP client with async support
- **urllib3**: Low-level HTTP functionality

### Development & Quality
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality

## MCP Server Ecosystem

The Elite Options System leverages a comprehensive suite of Model Context Protocol servers for enhanced capabilities:

### Caching & Performance
- **Redis MCP**: High-performance caching and persistence layer
  - Redis Server v7.0.15 running in Ubuntu WSL
  - Operations: set, get, delete, list (pattern matching)
  - Features: Real-time data caching, cross-session persistence
  - Performance: 60-70% faster dashboard loads, 75% faster analytics
  - Protocol: JSON-RPC 2.0 with WebSocket transport
  - Use Cases: Market data caching, trading pattern memory, signal persistence

### Database & Analytics
- **elite-options-database**: SQLite database operations
  - Operations: read_query, write_query, create_table, alter_table, drop_table
  - Analytics: export_query, list_tables, describe_table
  - Insights: append_insight, list_insights

### Market Intelligence
- **Hot News Server**: Real-time trending topics from 9 Chinese platforms
  - Platforms: Zhihu, 36Kr, Baidu, Bilibili, Weibo, Douyin, Hupu, Douban, IT News
  - Features: Heat metrics, trend analysis, sentiment indicators

### Research & Discovery
- **exa**: AI-powered search suite
  - Web search with content extraction
  - Research paper and academic content search
  - Company research and competitive analysis
  - LinkedIn professional network search
  - Wikipedia knowledge base access
  - GitHub repository and code search

- **Brave Search**: Web and local business search
  - General web search capabilities
  - Local business and location-based search
  - Content filtering and freshness controls

### Automation & Testing
- **Puppeteer**: Browser automation capabilities
  - Web scraping and data collection
  - Automated testing of dashboard functionality
  - Screenshot capture for documentation
  - Form filling and interaction simulation

### Knowledge Management
- **Persistent Knowledge Graph**: Relationship tracking and insights
  - Entity creation and management
  - Relationship mapping and analysis
  - Observation tracking and pattern recognition
  - Knowledge base evolution and learning

### Workflow & Process
- **TaskManager**: Systematic workflow management
  - Task planning and decomposition
  - Progress tracking and approval gates
  - Workflow orchestration and coordination

### Cognitive Enhancement
- **Sequential Thinking**: Structured reasoning frameworks
- **Memory**: Persistent context and learning systems
- **context7**: Advanced context management and analysis

## Development Setup

### Environment Management
- **Conda**: Primary environment management (recommended)
- **venv**: Alternative virtual environment option
- **requirements.txt**: Dependency specification and version pinning
- **environment.yml**: Conda environment specification

### Project Structure
```
elite_options_system_v2_5/
├── data_management/          # Data ingestion and processing
├── core_analytics_engine/    # Analytics pipeline and metrics
├── dashboard_application/    # Dash UI and visualization
├── data_models/             # Pydantic models and contracts
├── config/                  # Configuration management
├── tests/                   # Test suite
├── logs/                    # Application logging
├── cache/                   # Data and computation caching
└── memory-bank/             # Documentation and context
```

### Configuration Management
- **Environment Variables**: Sensitive configuration (API keys, endpoints)
- **Config Files**: Application settings and parameters
- **Pydantic Settings**: Type-safe configuration models
- **Multi-Environment**: Development, testing, production configurations

### Logging & Monitoring
- **Python logging**: Structured application logging
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Automatic log file management
- **Performance Monitoring**: Execution time and resource usage tracking

## Technical Constraints

### Platform Requirements
- **Operating System**: Windows 10+ (primary target)
- **Python Version**: 3.10 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage**: SSD recommended for data processing performance

### Data Source Limitations
- **Rate Limiting**: API endpoints have request limits
- **Availability**: Data sources may be temporarily unavailable
- **Data Quality**: Varying quality and completeness across sources
- **Latency**: Network delays affect real-time capabilities

### Performance Considerations
- **Memory Usage**: Large dataframes require careful memory management
- **Processing Time**: Complex analytics may take several seconds
- **UI Responsiveness**: Dashboard must remain interactive during processing
- **Concurrent Users**: Single-user application (no multi-tenancy)

### Security Constraints
- **API Keys**: Secure storage and rotation of credentials
- **Data Privacy**: No storage of sensitive trading information
- **Network Security**: HTTPS for all external communications
- **Local Storage**: Encrypted caching of sensitive data

## Dependencies

### Core Dependencies
```python
# Data Processing
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Visualization
plotly>=5.10.0
dash>=2.6.0
dash-bootstrap-components>=1.2.0

# Validation
pydantic>=1.10.0
typing-extensions>=4.3.0

# HTTP
requests>=2.28.0
aiohttp>=3.8.0
httpx>=0.23.0
```

### Development Dependencies
```python
# Testing
pytest>=7.1.0
pytest-asyncio>=0.19.0
pytest-cov>=3.0.0

# Code Quality
black>=22.6.0
flake8>=5.0.0
mypy>=0.971
pre-commit>=2.20.0
```

### Optional Dependencies
```python
# Performance
polars>=0.14.0  # Alternative to pandas
numba>=0.56.0   # JIT compilation

# Extended Analytics
scikit-learn>=1.1.0
statsmodels>=0.13.0
ta-lib>=0.4.0   # Technical analysis
```

## Integration Points

### External APIs
- **ConvexValue**: Primary options data source
- **Tradier**: Secondary options and equity data
- **Yahoo Finance**: Fallback data source
- **MCP Servers**: Enhanced capabilities through protocol integration

### Data Formats
- **JSON**: API responses and configuration
- **CSV**: Data export and import
- **Parquet**: Efficient data storage and caching
- **SQLite**: Local database storage (via MCP)

### Communication Protocols
- **HTTP/HTTPS**: API communication
- **WebSocket**: Real-time data updates (future)
- **MCP Protocol**: Model Context Protocol for server integration
- **File System**: Local data and configuration storage

## Deployment Considerations

### Local Development
- **Hot Reload**: Automatic code reloading during development
- **Debug Mode**: Enhanced error reporting and logging
- **Test Data**: Synthetic data for development and testing

### Production Deployment
- **Process Management**: Systemd or Windows Service
- **Reverse Proxy**: Nginx for production web serving
- **SSL/TLS**: Certificate management for secure connections
- **Backup Strategy**: Data and configuration backup procedures

### Monitoring & Maintenance
- **Health Checks**: System status monitoring
- **Performance Metrics**: Response time and resource usage
- **Error Tracking**: Automated error reporting and alerting
- **Update Management**: Dependency updates and security patches