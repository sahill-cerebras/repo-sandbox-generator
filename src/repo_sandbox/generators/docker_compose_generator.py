"""Docker Compose generation module."""

import logging
from typing import Dict, Any, List
from jinja2 import Template

logger = logging.getLogger(__name__)


class DockerComposeGenerator:
    """Generates docker-compose.yml files based on repository analysis."""
    
    COMPOSE_TEMPLATE = """
version: '3.8'

services:
  {{ service_name }}:
    build: .
    container_name: {{ container_name }}
    {% if ports %}
    ports:
    {% for port_mapping in ports %}
      - "{{ port_mapping }}"
    {% endfor %}
    {% endif %}
    {% if volumes %}
    volumes:
    {% for volume in volumes %}
      - {{ volume }}
    {% endfor %}
    {% endif %}
    {% if environment_vars %}
    environment:
    {% for key, value in environment_vars.items() %}
      {{ key }}: {{ value }}
    {% endfor %}
    {% endif %}
    {% if depends_on %}
    depends_on:
    {% for service in depends_on %}
      - {{ service }}
    {% endfor %}
    {% endif %}
    {% if command %}
    command: {{ command }}
    {% endif %}
    {% if networks %}
    networks:
    {% for network in networks %}
      - {{ network }}
    {% endfor %}
    {% endif %}

{% if additional_services %}
{% for service_name, service_config in additional_services.items() %}
  {{ service_name }}:
    {% for key, value in service_config.items() %}
    {{ key }}: {{ value }}
    {% endfor %}

{% endfor %}
{% endif %}

{% if volumes_definition %}
volumes:
{% for volume_name, volume_config in volumes_definition.items() %}
  {{ volume_name }}:
    {% if volume_config %}
    {% for key, value in volume_config.items() %}
    {{ key }}: {{ value }}
    {% endfor %}
    {% endif %}

{% endfor %}
{% endif %}

{% if networks_definition %}
networks:
{% for network_name, network_config in networks_definition.items() %}
  {{ network_name }}:
    {% if network_config %}
    {% for key, value in network_config.items() %}
    {{ key }}: {{ value }}
    {% endfor %}
    {% endif %}

{% endfor %}
{% endif %}
""".strip()

    def generate_docker_compose(self, analysis: Dict[str, Any]) -> str:
        """
        Generate docker-compose.yml content based on analysis.
        
        Args:
            analysis: Repository analysis results
            
        Returns:
            Generated docker-compose.yml content
        """
        repo_name = analysis['repo_name']
        
        # Build base service configuration
        service_name = self._sanitize_service_name(repo_name)
        
        template_vars = {
            'service_name': service_name,
            'container_name': f"{service_name}_app",
            'ports': self._generate_port_mappings(analysis),
            'volumes': self._generate_volumes(analysis),
            'environment_vars': self._generate_environment_vars(analysis),
            'command': self._generate_command(analysis),
            'depends_on': [],
            'networks': [],
            'additional_services': {},
            'volumes_definition': {},
            'networks_definition': {}
        }
        
        # Render template
        jinja_template = Template(self.COMPOSE_TEMPLATE)
        compose_content = jinja_template.render(**template_vars)
        
        logger.info(f"Generated docker-compose.yml for {service_name}")
        return compose_content
    
    def _sanitize_service_name(self, name: str) -> str:
        """Sanitize service name for docker-compose."""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or 'app'
    
    def _generate_port_mappings(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate port mappings for the main service."""
        ports = analysis.get('expose_ports', [])
        if not ports:
            return []
        
        port_mappings = []
        for port in ports:
            port_mappings.append(f"{port}:{port}")
        
        return port_mappings
    
    def _generate_volumes(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate volume mappings."""
        volumes = [
            ".:/app",  # Mount source code for development
        ]
        
    # Framework-specific volumes removed (framework detection disabled)
        
        return volumes
    
    def _generate_environment_vars(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate environment variables for docker-compose."""
        env_vars = analysis.get('environment_vars', {}).copy()
        
        # Add development-friendly defaults
        env_vars.update({
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
        })
        
    # Framework-specific environment variables removed
        
        return env_vars
    
    def _generate_command(self, analysis: Dict[str, Any]) -> str:
        """Generate command for the main service."""
        # Return None to use the default command from Dockerfile
        return None
    
    # Framework service helpers removed (framework detection disabled)
    
    def _add_database_service(self, template_vars: Dict[str, Any], service_name: str):
        """Add PostgreSQL database service."""
        db_service = {
            'image': 'postgres:13',
            'container_name': f'{service_name}_db',
            'environment': {
                'POSTGRES_DB': service_name,
                'POSTGRES_USER': 'postgres',
                'POSTGRES_PASSWORD': 'postgres'
            },
            'volumes': ['postgres_data:/var/lib/postgresql/data'],
            'ports': ['5432:5432'],
            'networks': [f'{service_name}_network']
        }
        
        template_vars['additional_services']['db'] = db_service
        template_vars['depends_on'].append('db')
        
        # Add database URL to main service
        template_vars['environment_vars']['DATABASE_URL'] = (
            f'postgresql://postgres:postgres@db:5432/{service_name}'
        )
    
    def _add_redis_service(self, template_vars: Dict[str, Any], service_name: str):
        """Add Redis service."""
        redis_service = {
            'image': 'redis:6-alpine',
            'container_name': f'{service_name}_redis',
            'volumes': ['redis_data:/data'],
            'ports': ['6379:6379'],
            'networks': [f'{service_name}_network']
        }
        
        template_vars['additional_services']['redis'] = redis_service
        
        # Add Redis URL to main service
        template_vars['environment_vars']['REDIS_URL'] = 'redis://redis:6379/0'
