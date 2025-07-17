# Lead Enrichment Platform

A professional-grade web application that enriches lead data by scraping LinkedIn profiles and generating personalized outreach messages using AI.

## Features

- **Smart Column Detection**: Automatically detects and suggests column mappings for names, LinkedIn profiles, and company URLs
- **Customizable AI Prompts**: Users can customize the AI prompt template for generating personalized messages
- **Flexible Configuration**: Users can provide their own LinkedIn credentials and API keys
- **Rate Limiting**: Built-in protection against abuse with configurable rate limits
- **Real-time Progress**: Live progress tracking with detailed status updates
- **Production Ready**: Dockerized with nginx reverse proxy and health checks
- **Security First**: Input validation, secure file handling, and comprehensive error handling

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd lead-enrichment-platform
```

2. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the application:
```bash
docker-compose up -d
```

4. Access the application at `http://localhost`

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Chrome and ChromeDriver:
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install google-chrome-stable
wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/LATEST_RELEASE/chromedriver_linux64.zip
unzip /tmp/chromedriver.zip -d /usr/local/bin/
chmod +x /usr/local/bin/chromedriver
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
python app.py
```

## Configuration

### Environment Variables

- `SECRET_KEY`: Flask secret key for session security
- `GEMINI_API_KEY`: Default Gemini API key (users can override)
- `CHROMEDRIVER_PATH`: Path to ChromeDriver executable
- `FLASK_ENV`: Set to 'production' for production deployment

### User Configuration Options

Users can configure the following through the web interface:

- **LinkedIn Credentials**: Email and password for LinkedIn access
- **Gemini API Key**: Personal API key for better rate limits
- **Custom Prompts**: Personalized AI prompt templates
- **Processing Delay**: Time between profile scraping (minimum 8 seconds)
- **Column Mapping**: Map CSV columns to required fields

## API Endpoints

- `GET /`: Main application interface
- `POST /analyze-csv`: Analyze CSV structure and suggest mappings
- `POST /upload`: Upload CSV and start processing
- `GET /status/<session_id>`: Get processing status
- `GET /download/<filename>`: Download enriched CSV
- `GET /health`: Health check endpoint

## Security Features

- Rate limiting (200 requests/day, 50/hour)
- File size limits (16MB maximum)
- Input validation and sanitization
- Secure filename handling
- Session-based file isolation
- CSRF protection
- Security headers via nginx

## Production Deployment

### Docker Deployment

The application includes production-ready Docker configuration:

```bash
# Build and deploy
docker-compose up -d

# Scale workers
docker-compose up -d --scale lead-enrichment=3

# View logs
docker-compose logs -f
```

### Manual Deployment

For manual deployment, use a production WSGI server:

```bash
# Using Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 app:app

# Using uWSGI
uwsgi --http :5000 --module app:app --processes 4 --threads 2
```

### Nginx Configuration

The included nginx configuration provides:
- Reverse proxy with load balancing
- Rate limiting
- Security headers
- Gzip compression
- SSL termination (configure certificates)

## Monitoring

### Health Checks

The application provides a health check endpoint:
```bash
curl http://localhost:5000/health
```

### Logging

Application logs include:
- Request/response logging
- Error tracking
- Processing status updates
- Security events

### Metrics

Monitor these key metrics:
- Active processing sessions
- Request rates and response times
- Error rates
- Resource usage (CPU, memory)

## Troubleshooting

### Common Issues

1. **ChromeDriver Issues**:
   - Ensure ChromeDriver version matches Chrome version
   - Check ChromeDriver path in environment variables
   - Verify Chrome is installed in Docker container

2. **LinkedIn Access**:
   - Use valid LinkedIn credentials
   - Respect rate limits to avoid blocking
   - Consider using multiple accounts for higher throughput

3. **API Rate Limits**:
   - Provide personal Gemini API key for better limits
   - Adjust processing delay to reduce API calls
   - Monitor API usage in Google Cloud Console

### Performance Optimization

- Use SSD storage for faster file I/O
- Increase worker processes for higher throughput
- Configure Redis for session storage in multi-instance deployments
- Use CDN for static assets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review application logs for error details