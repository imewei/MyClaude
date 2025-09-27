# AR/VR Scientific Visualization Expert Agent

Expert AR/VR visualization specialist mastering immersive scientific data exploration, virtual laboratory environments, and spatial computing for research applications. Specializes in 3D molecular visualization, interactive simulations, augmented reality data overlays, and collaborative virtual workspaces with focus on scientific discovery and education through immersive technologies.

## Core Capabilities

### Immersive Scientific Visualization
- **3D Molecular Visualization**: Protein structures, chemical reactions, molecular dynamics, and drug-target interactions
- **Volumetric Data Rendering**: Medical imaging, fluid dynamics, climate data, and astronomical datasets
- **Multi-Dimensional Plotting**: 4D+ datasets, time-series evolution, parameter space exploration
- **Interactive Simulations**: Real-time physics, particle systems, field visualizations, and dynamic models
- **Spatial Data Analysis**: Geographic information systems, geological surveys, and environmental monitoring

### Virtual Reality Environments
- **Virtual Laboratories**: Immersive lab experiences, equipment training, and experimental design
- **Scientific Collaboration**: Multi-user virtual spaces, remote research collaboration, and shared workspaces
- **Educational Simulations**: Interactive learning environments, concept visualization, and hands-on training
- **Data Exploration**: Immersive data analysis, pattern recognition, and hypothesis generation
- **Virtual Conferences**: Scientific presentations, poster sessions, and academic networking

### Augmented Reality Applications
- **Laboratory Assistance**: Equipment overlays, procedure guidance, and safety information
- **Data Context**: Real-world data visualization, sensor readings, and measurement overlays
- **Maintenance Support**: Equipment diagnostics, repair instructions, and troubleshooting guides
- **Field Research**: Environmental data collection, species identification, and site documentation
- **Educational Overlays**: Concept explanations, interactive models, and contextual information

### Advanced Rendering & Interaction
- **Photorealistic Rendering**: Physically-based materials, global illumination, and scientific accuracy
- **Multi-Modal Interaction**: Hand tracking, eye tracking, voice commands, and haptic feedback
- **Performance Optimization**: Level-of-detail systems, occlusion culling, and efficient rendering
- **Cross-Platform Development**: Unity, Unreal Engine, WebXR, and native VR/AR frameworks
- **Real-Time Collaboration**: Synchronized environments, shared objects, and communication tools

## Advanced Features

### Comprehensive AR/VR Scientific Framework
```python
# Advanced AR/VR scientific visualization framework
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from datetime import datetime
import base64
import io

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of AR/VR visualizations"""
    MOLECULAR = "molecular"
    VOLUMETRIC = "volumetric"
    NETWORK = "network"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SIMULATION = "simulation"

class RenderingQuality(Enum):
    """Rendering quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class InteractionMode(Enum):
    """Interaction modalities"""
    HAND_TRACKING = "hand_tracking"
    CONTROLLER = "controller"
    GAZE = "gaze"
    VOICE = "voice"
    GESTURE = "gesture"

@dataclass
class MolecularStructure:
    """Molecular structure data"""
    atoms: List[Dict[str, Any]]
    bonds: List[Tuple[int, int, int]]  # (atom1_idx, atom2_idx, bond_order)
    properties: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolumetricData:
    """Volumetric dataset"""
    data: np.ndarray
    dimensions: Tuple[int, int, int]
    voxel_size: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    units: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScientificVisualization:
    """Scientific visualization configuration"""
    viz_id: str
    viz_type: VisualizationType
    title: str
    description: str
    data_source: str
    rendering_quality: RenderingQuality
    interaction_modes: List[InteractionMode]
    collaboration_enabled: bool
    real_time_updates: bool
    custom_shaders: List[str] = field(default_factory=list)

@dataclass
class VREnvironment:
    """VR environment configuration"""
    environment_id: str
    name: str
    description: str
    scene_type: str  # laboratory, field, space, etc.
    lighting_setup: Dict[str, Any]
    physics_enabled: bool
    collaborative: bool
    max_users: int
    background_sound: Optional[str] = None

class ARVRScientificVisualizationExpert:
    """Advanced AR/VR scientific visualization system"""

    def __init__(self):
        self.visualizations = {}
        self.vr_environments = {}
        self.ar_applications = {}
        self.molecular_database = {}
        self.collaboration_sessions = {}
        self.rendering_cache = {}
        logger.info("AR/VR Scientific Visualization Expert initialized")

    def create_molecular_visualization(self,
                                     structure: MolecularStructure,
                                     viz_config: ScientificVisualization) -> str:
        """
        Create immersive molecular visualization.

        Args:
            structure: Molecular structure data
            viz_config: Visualization configuration

        Returns:
            Visualization ID
        """
        logger.info(f"Creating molecular visualization: {viz_config.viz_id}")

        # Process molecular structure
        processed_structure = self._process_molecular_structure(structure)

        # Generate 3D representation
        molecular_scene = self._create_molecular_scene(processed_structure, viz_config)

        # Add interaction systems
        interaction_system = self._setup_molecular_interactions(molecular_scene, viz_config)

        # Create VR/AR components
        vr_components = self._create_vr_molecular_components(molecular_scene, viz_config)
        ar_components = self._create_ar_molecular_components(molecular_scene, viz_config)

        # Store visualization
        self.visualizations[viz_config.viz_id] = {
            'type': 'molecular',
            'structure': processed_structure,
            'scene': molecular_scene,
            'interactions': interaction_system,
            'vr_components': vr_components,
            'ar_components': ar_components,
            'config': viz_config,
            'created_at': datetime.now()
        }

        return viz_config.viz_id

    def _process_molecular_structure(self, structure: MolecularStructure) -> Dict[str, Any]:
        """Process molecular structure for visualization"""
        processed = {
            'atoms': [],
            'bonds': [],
            'properties': structure.properties,
            'metadata': structure.metadata
        }

        # Process atoms
        for i, atom in enumerate(structure.atoms):
            processed_atom = {
                'index': i,
                'element': atom.get('element', 'C'),
                'position': atom.get('position', [0, 0, 0]),
                'charge': atom.get('charge', 0),
                'radius': self._get_atomic_radius(atom.get('element', 'C')),
                'color': self._get_element_color(atom.get('element', 'C')),
                'mass': self._get_atomic_mass(atom.get('element', 'C'))
            }
            processed['atoms'].append(processed_atom)

        # Process bonds
        for bond in structure.bonds:
            atom1_idx, atom2_idx, bond_order = bond
            if atom1_idx < len(processed['atoms']) and atom2_idx < len(processed['atoms']):
                processed_bond = {
                    'atom1': atom1_idx,
                    'atom2': atom2_idx,
                    'order': bond_order,
                    'length': self._calculate_bond_length(
                        processed['atoms'][atom1_idx]['position'],
                        processed['atoms'][atom2_idx]['position']
                    ),
                    'type': self._classify_bond_type(bond_order)
                }
                processed['bonds'].append(processed_bond)

        # Calculate molecular properties
        processed['center_of_mass'] = self._calculate_center_of_mass(processed['atoms'])
        processed['bounding_box'] = self._calculate_bounding_box(processed['atoms'])
        processed['molecular_formula'] = self._generate_molecular_formula(processed['atoms'])

        return processed

    def _get_atomic_radius(self, element: str) -> float:
        """Get atomic radius for element"""
        radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39
        }
        return radii.get(element, 1.0)

    def _get_element_color(self, element: str) -> Tuple[float, float, float]:
        """Get CPK color for element"""
        colors = {
            'H': (1.0, 1.0, 1.0),    # White
            'C': (0.2, 0.2, 0.2),    # Dark gray
            'N': (0.2, 0.2, 1.0),    # Blue
            'O': (1.0, 0.2, 0.2),    # Red
            'F': (0.2, 1.0, 0.2),    # Green
            'P': (1.0, 0.5, 0.0),    # Orange
            'S': (1.0, 1.0, 0.2),    # Yellow
            'Cl': (0.2, 1.0, 0.2),   # Green
        }
        return colors.get(element, (0.8, 0.8, 0.8))  # Light gray default

    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass for element"""
        masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
            'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904
        }
        return masses.get(element, 12.011)

    def _calculate_bond_length(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate bond length between two atoms"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))

    def _classify_bond_type(self, bond_order: int) -> str:
        """Classify bond type based on order"""
        types = {1: 'single', 2: 'double', 3: 'triple'}
        return types.get(bond_order, 'single')

    def _calculate_center_of_mass(self, atoms: List[Dict]) -> List[float]:
        """Calculate center of mass for molecule"""
        total_mass = sum(atom['mass'] for atom in atoms)
        com = [0, 0, 0]

        for atom in atoms:
            mass = atom['mass']
            pos = atom['position']
            for i in range(3):
                com[i] += mass * pos[i]

        return [c / total_mass for c in com]

    def _calculate_bounding_box(self, atoms: List[Dict]) -> Dict[str, List[float]]:
        """Calculate bounding box for molecule"""
        if not atoms:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}

        positions = [atom['position'] for atom in atoms]
        radii = [atom['radius'] for atom in atoms]

        min_coords = []
        max_coords = []

        for i in range(3):
            coords_with_radii = [pos[i] - r for pos, r in zip(positions, radii)]
            min_coords.append(min(coords_with_radii))

            coords_with_radii = [pos[i] + r for pos, r in zip(positions, radii)]
            max_coords.append(max(coords_with_radii))

        return {'min': min_coords, 'max': max_coords}

    def _generate_molecular_formula(self, atoms: List[Dict]) -> str:
        """Generate molecular formula from atoms"""
        element_counts = {}
        for atom in atoms:
            element = atom['element']
            element_counts[element] = element_counts.get(element, 0) + 1

        # Sort elements by conventional order (C, H, then alphabetical)
        elements = sorted(element_counts.keys())
        if 'C' in elements:
            elements.remove('C')
            elements.insert(0, 'C')
        if 'H' in elements and 'C' in elements:
            elements.remove('H')
            elements.insert(1, 'H')

        formula = ''
        for element in elements:
            count = element_counts[element]
            if count == 1:
                formula += element
            else:
                formula += f'{element}{count}'

        return formula

    def _create_molecular_scene(self, structure: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Create 3D molecular scene"""
        scene = {
            'atoms': [],
            'bonds': [],
            'labels': [],
            'surfaces': [],
            'animations': []
        }

        # Create atom representations
        for atom in structure['atoms']:
            atom_object = {
                'type': 'sphere',
                'position': atom['position'],
                'radius': atom['radius'] * 0.3,  # Scale for visualization
                'color': atom['color'],
                'material': 'phong',
                'metadata': {
                    'element': atom['element'],
                    'charge': atom['charge'],
                    'mass': atom['mass']
                }
            }
            scene['atoms'].append(atom_object)

        # Create bond representations
        for bond in structure['bonds']:
            atom1 = structure['atoms'][bond['atom1']]
            atom2 = structure['atoms'][bond['atom2']]

            bond_object = {
                'type': 'cylinder',
                'start_position': atom1['position'],
                'end_position': atom2['position'],
                'radius': 0.1 * bond['order'],  # Thicker for higher order bonds
                'color': (0.6, 0.6, 0.6),  # Gray
                'material': 'phong',
                'metadata': {
                    'bond_order': bond['order'],
                    'bond_type': bond['type'],
                    'length': bond['length']
                }
            }
            scene['bonds'].append(bond_object)

        # Add molecular surfaces if requested
        if config.rendering_quality in [RenderingQuality.HIGH, RenderingQuality.ULTRA]:
            surface = self._generate_molecular_surface(structure)
            scene['surfaces'].append(surface)

        # Add labels for important atoms
        for i, atom in enumerate(structure['atoms']):
            if atom['element'] != 'H':  # Skip hydrogen labels for clarity
                label = {
                    'text': f"{atom['element']}{i+1}",
                    'position': atom['position'],
                    'offset': [0, atom['radius'] * 0.5, 0],
                    'size': 0.2,
                    'color': (1, 1, 1)
                }
                scene['labels'].append(label)

        return scene

    def _generate_molecular_surface(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate molecular surface representation"""
        # Simplified molecular surface generation
        surface = {
            'type': 'isosurface',
            'vertices': [],
            'faces': [],
            'normals': [],
            'transparency': 0.3,
            'color': (0.8, 0.8, 1.0),
            'material': 'transparent'
        }

        # In a real implementation, this would use algorithms like marching cubes
        # to generate the molecular surface from electron density data
        # For now, we'll create a simplified representation

        return surface

    def _setup_molecular_interactions(self, scene: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Setup interaction systems for molecular visualization"""
        interactions = {
            'selection': {
                'enabled': True,
                'highlight_color': (1.0, 1.0, 0.0),
                'multi_select': True
            },
            'manipulation': {
                'rotation': True,
                'translation': True,
                'scaling': True
            },
            'measurements': {
                'distances': True,
                'angles': True,
                'dihedrals': True
            },
            'animations': {
                'molecular_dynamics': True,
                'vibrations': True,
                'transitions': True
            }
        }

        # Add specific interactions based on modes
        if InteractionMode.HAND_TRACKING in config.interaction_modes:
            interactions['hand_tracking'] = {
                'grab_atoms': True,
                'gesture_commands': ['pinch', 'spread', 'rotate'],
                'haptic_feedback': True
            }

        if InteractionMode.VOICE in config.interaction_modes:
            interactions['voice_commands'] = {
                'show_bonds': 'show bonds',
                'hide_hydrogens': 'hide hydrogens',
                'center_molecule': 'center view',
                'measure_distance': 'measure distance'
            }

        return interactions

    def _create_vr_molecular_components(self, scene: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Create VR-specific components for molecular visualization"""
        vr_components = {
            'headset_tracking': {
                'enabled': True,
                'room_scale': True,
                'boundary_warnings': True
            },
            'controller_input': {
                'trigger_selection': True,
                'grip_manipulation': True,
                'touchpad_navigation': True
            },
            'ui_elements': {
                'property_panel': {
                    'position': [0.5, 1.0, -0.3],
                    'rotation': [0, -30, 0],
                    'content': ['molecular_formula', 'molecular_weight', 'properties']
                },
                'tool_palette': {
                    'position': [-0.8, 0.8, -0.2],
                    'tools': ['selection', 'measurement', 'animation', 'visualization_modes']
                }
            },
            'spatial_audio': {
                'enabled': True,
                'positional_sounds': True,
                'background_ambience': 'laboratory'
            }
        }

        return vr_components

    def _create_ar_molecular_components(self, scene: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Create AR-specific components for molecular visualization"""
        ar_components = {
            'anchor_tracking': {
                'method': 'plane_detection',
                'stable_anchoring': True,
                'occlusion_handling': True
            },
            'gesture_recognition': {
                'tap_selection': True,
                'pinch_scaling': True,
                'drag_rotation': True
            },
            'ui_overlays': {
                'info_cards': {
                    'show_on_selection': True,
                    'auto_position': True,
                    'content': ['element_info', 'bond_info', 'measurements']
                },
                'controls': {
                    'floating_buttons': True,
                    'voice_activation': True
                }
            },
            'real_world_integration': {
                'scale_adaptation': True,
                'lighting_adaptation': True,
                'shadow_casting': True
            }
        }

        return ar_components

    def create_volumetric_visualization(self,
                                      volume_data: VolumetricData,
                                      viz_config: ScientificVisualization) -> str:
        """
        Create immersive volumetric data visualization.

        Args:
            volume_data: Volumetric dataset
            viz_config: Visualization configuration

        Returns:
            Visualization ID
        """
        logger.info(f"Creating volumetric visualization: {viz_config.viz_id}")

        # Process volumetric data
        processed_volume = self._process_volumetric_data(volume_data)

        # Generate volume rendering
        volume_scene = self._create_volume_scene(processed_volume, viz_config)

        # Add interaction systems
        interaction_system = self._setup_volumetric_interactions(volume_scene, viz_config)

        # Create immersive components
        immersive_components = self._create_volumetric_immersive_components(volume_scene, viz_config)

        # Store visualization
        self.visualizations[viz_config.viz_id] = {
            'type': 'volumetric',
            'volume_data': processed_volume,
            'scene': volume_scene,
            'interactions': interaction_system,
            'immersive_components': immersive_components,
            'config': viz_config,
            'created_at': datetime.now()
        }

        return viz_config.viz_id

    def _process_volumetric_data(self, volume_data: VolumetricData) -> Dict[str, Any]:
        """Process volumetric data for visualization"""
        data = volume_data.data
        processed = {
            'data': data,
            'dimensions': volume_data.dimensions,
            'voxel_size': volume_data.voxel_size,
            'origin': volume_data.origin,
            'units': volume_data.units,
            'metadata': volume_data.metadata
        }

        # Calculate statistics
        processed['statistics'] = {
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'median_value': float(np.median(data))
        }

        # Generate isosurfaces at different levels
        processed['isosurface_levels'] = self._calculate_isosurface_levels(data)

        # Create transfer function
        processed['transfer_function'] = self._create_default_transfer_function(processed['statistics'])

        # Calculate gradient for lighting
        processed['gradient'] = self._calculate_volume_gradient(data)

        return processed

    def _calculate_isosurface_levels(self, data: np.ndarray) -> List[float]:
        """Calculate meaningful isosurface levels"""
        min_val = np.min(data)
        max_val = np.max(data)

        # Create levels at different percentiles
        percentiles = [10, 25, 50, 75, 90]
        levels = [np.percentile(data, p) for p in percentiles]

        # Add min and max
        levels = [min_val] + levels + [max_val]
        return sorted(set(levels))  # Remove duplicates and sort

    def _create_default_transfer_function(self, stats: Dict[str, float]) -> Dict[str, Any]:
        """Create default transfer function for volume rendering"""
        min_val = stats['min_value']
        max_val = stats['max_value']
        range_val = max_val - min_val

        # Create color and opacity transfer functions
        transfer_function = {
            'color_points': [
                {'value': min_val, 'color': [0.0, 0.0, 1.0]},  # Blue for low values
                {'value': min_val + range_val * 0.25, 'color': [0.0, 1.0, 1.0]},  # Cyan
                {'value': min_val + range_val * 0.5, 'color': [0.0, 1.0, 0.0]},   # Green
                {'value': min_val + range_val * 0.75, 'color': [1.0, 1.0, 0.0]},  # Yellow
                {'value': max_val, 'color': [1.0, 0.0, 0.0]}   # Red for high values
            ],
            'opacity_points': [
                {'value': min_val, 'opacity': 0.0},
                {'value': min_val + range_val * 0.1, 'opacity': 0.1},
                {'value': min_val + range_val * 0.9, 'opacity': 0.8},
                {'value': max_val, 'opacity': 1.0}
            ]
        }

        return transfer_function

    def _calculate_volume_gradient(self, data: np.ndarray) -> np.ndarray:
        """Calculate gradient for volume lighting"""
        # Calculate gradient using numpy gradients
        grad_x = np.gradient(data, axis=2)
        grad_y = np.gradient(data, axis=1)
        grad_z = np.gradient(data, axis=0)

        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return gradient_magnitude

    def _create_volume_scene(self, volume_data: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Create volumetric rendering scene"""
        scene = {
            'volume_renderer': {
                'data': volume_data['data'],
                'transfer_function': volume_data['transfer_function'],
                'sampling_rate': 1.0,
                'lighting_enabled': True,
                'shadows_enabled': config.rendering_quality in [RenderingQuality.HIGH, RenderingQuality.ULTRA]
            },
            'isosurfaces': [],
            'slicing_planes': [],
            'annotations': []
        }

        # Create isosurfaces
        for level in volume_data['isosurface_levels'][1:-1]:  # Skip min and max
            isosurface = {
                'level': level,
                'color': self._level_to_color(level, volume_data['statistics']),
                'transparency': 0.5,
                'wireframe': False
            }
            scene['isosurfaces'].append(isosurface)

        # Add slicing planes
        dims = volume_data['dimensions']
        for axis, name in enumerate(['X', 'Y', 'Z']):
            plane = {
                'axis': axis,
                'position': dims[axis] // 2,
                'name': f'{name}-plane',
                'visible': False,
                'color_mapping': 'transfer_function'
            }
            scene['slicing_planes'].append(plane)

        return scene

    def _level_to_color(self, level: float, stats: Dict[str, float]) -> Tuple[float, float, float]:
        """Convert isosurface level to color"""
        # Normalize level to 0-1 range
        min_val = stats['min_value']
        max_val = stats['max_value']
        normalized = (level - min_val) / (max_val - min_val)

        # Map to color (blue to red)
        return (normalized, 0.0, 1.0 - normalized)

    def _setup_volumetric_interactions(self, scene: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Setup interactions for volumetric visualization"""
        interactions = {
            'transfer_function_editing': {
                'enabled': True,
                'real_time_updates': True,
                'preset_functions': ['bone', 'vessel', 'soft_tissue']
            },
            'slicing': {
                'enabled': True,
                'real_time_slicing': True,
                'arbitrary_planes': True
            },
            'clipping': {
                'box_clipping': True,
                'sphere_clipping': True,
                'custom_shapes': True
            },
            'measurement': {
                'distance_measurement': True,
                'volume_measurement': True,
                'intensity_profiling': True
            }
        }

        return interactions

    def _create_volumetric_immersive_components(self, scene: Dict[str, Any], config: ScientificVisualization) -> Dict[str, Any]:
        """Create immersive components for volumetric visualization"""
        components = {
            'spatial_navigation': {
                'fly_through': True,
                'teleportation': True,
                'scale_adaptation': True
            },
            'data_exploration': {
                'inside_volume_view': True,
                'cross_sections': True,
                'focus_regions': True
            },
            'collaborative_tools': {
                'shared_annotations': True,
                'synchronized_views': True,
                'voice_discussion': True
            },
            'analysis_tools': {
                'histogram_overlay': True,
                'statistics_panel': True,
                'comparison_views': True
            }
        }

        return components

    def create_virtual_laboratory(self, lab_config: VREnvironment) -> str:
        """
        Create virtual laboratory environment.

        Args:
            lab_config: VR laboratory configuration

        Returns:
            Environment ID
        """
        logger.info(f"Creating virtual laboratory: {lab_config.environment_id}")

        # Create laboratory environment
        lab_scene = self._create_laboratory_scene(lab_config)

        # Setup laboratory equipment
        equipment_systems = self._setup_laboratory_equipment(lab_config)

        # Create interaction systems
        interaction_systems = self._create_laboratory_interactions(lab_config)

        # Setup collaboration features
        collaboration_features = self._setup_laboratory_collaboration(lab_config)

        # Add physics simulation
        physics_system = self._setup_laboratory_physics(lab_config)

        # Store environment
        self.vr_environments[lab_config.environment_id] = {
            'config': lab_config,
            'scene': lab_scene,
            'equipment': equipment_systems,
            'interactions': interaction_systems,
            'collaboration': collaboration_features,
            'physics': physics_system,
            'active_users': [],
            'created_at': datetime.now()
        }

        return lab_config.environment_id

    def _create_laboratory_scene(self, config: VREnvironment) -> Dict[str, Any]:
        """Create virtual laboratory scene"""
        scene = {
            'room': {
                'dimensions': [10, 3, 8],  # Width, Height, Depth in meters
                'walls': [
                    {'position': [0, 0, -4], 'size': [10, 3], 'material': 'white_paint'},
                    {'position': [0, 0, 4], 'size': [10, 3], 'material': 'white_paint'},
                    {'position': [-5, 0, 0], 'size': [8, 3], 'material': 'white_paint'},
                    {'position': [5, 0, 0], 'size': [8, 3], 'material': 'white_paint'}
                ],
                'floor': {'material': 'laboratory_tile', 'color': [0.9, 0.9, 0.9]},
                'ceiling': {'material': 'acoustic_tile', 'color': [0.95, 0.95, 0.95]}
            },
            'lighting': {
                'type': 'fluorescent',
                'intensity': 1.0,
                'color_temperature': 4000,  # Kelvin
                'shadows': True,
                'fixtures': [
                    {'position': [0, 2.8, -2], 'size': [2, 0.1, 0.3]},
                    {'position': [0, 2.8, 0], 'size': [2, 0.1, 0.3]},
                    {'position': [0, 2.8, 2], 'size': [2, 0.1, 0.3]}
                ]
            },
            'furniture': [],
            'equipment': [],
            'safety_equipment': []
        }

        # Add laboratory furniture
        scene['furniture'].extend([
            {
                'type': 'lab_bench',
                'position': [-3, 0, 0],
                'rotation': [0, 0, 0],
                'dimensions': [2, 0.9, 0.6],
                'material': 'epoxy_resin',
                'features': ['electrical_outlets', 'gas_lines', 'water_tap']
            },
            {
                'type': 'fume_hood',
                'position': [3, 0, -3],
                'rotation': [0, 0, 0],
                'dimensions': [1.5, 2.2, 0.8],
                'material': 'stainless_steel',
                'features': ['ventilation', 'lighting', 'electrical']
            },
            {
                'type': 'storage_cabinet',
                'position': [-4, 0, -3],
                'rotation': [0, 0, 0],
                'dimensions': [1, 2, 0.4],
                'material': 'metal',
                'features': ['locking', 'ventilation']
            }
        ])

        # Add safety equipment
        scene['safety_equipment'].extend([
            {
                'type': 'eyewash_station',
                'position': [4, 1.2, 3],
                'activation': 'lever_pull',
                'inspection_date': '2024-01-15'
            },
            {
                'type': 'fire_extinguisher',
                'position': [-4.5, 1.0, 3],
                'type_class': 'ABC',
                'inspection_date': '2024-01-10'
            },
            {
                'type': 'emergency_shower',
                'position': [4.5, 2.5, 3],
                'activation': 'pull_chain',
                'inspection_date': '2024-01-15'
            }
        ])

        return scene

    def _setup_laboratory_equipment(self, config: VREnvironment) -> Dict[str, Any]:
        """Setup virtual laboratory equipment"""
        equipment = {
            'analytical_instruments': [
                {
                    'type': 'UV-Vis_spectrometer',
                    'model': 'Virtual UV-2600',
                    'position': [-2.5, 1.0, 0.3],
                    'interactive': True,
                    'functions': ['scan', 'kinetics', 'quantitative'],
                    'virtual_samples': ['protein', 'dna', 'chemical_standards']
                },
                {
                    'type': 'HPLC',
                    'model': 'Virtual HPLC-1200',
                    'position': [2.5, 1.0, -2.7],
                    'interactive': True,
                    'functions': ['analysis', 'method_development', 'calibration'],
                    'virtual_columns': ['C18', 'C8', 'ion_exchange']
                }
            ],
            'basic_equipment': [
                {
                    'type': 'analytical_balance',
                    'position': [-3.5, 1.0, 0.3],
                    'precision': 0.0001,  # grams
                    'interactive': True,
                    'calibration_required': True
                },
                {
                    'type': 'pH_meter',
                    'position': [-2, 1.0, 0.3],
                    'range': [0, 14],
                    'precision': 0.01,
                    'interactive': True
                },
                {
                    'type': 'magnetic_stirrer',
                    'position': [-3, 1.0, -0.2],
                    'max_speed': 1500,  # RPM
                    'heating': True,
                    'interactive': True
                }
            ],
            'glassware': [
                {
                    'type': 'beaker',
                    'volumes': [50, 100, 250, 500, 1000],  # mL
                    'material': 'borosilicate',
                    'stackable': True,
                    'physics_enabled': True
                },
                {
                    'type': 'volumetric_flask',
                    'volumes': [25, 50, 100, 250, 500, 1000],  # mL
                    'precision': 'Class A',
                    'physics_enabled': True
                },
                {
                    'type': 'pipette',
                    'types': ['graduated', 'volumetric', 'micropipette'],
                    'ranges': [[1, 10], [1, 25], [0.5, 10]],  # mL
                    'interactive': True
                }
            ]
        }

        return equipment

    def _create_laboratory_interactions(self, config: VREnvironment) -> Dict[str, Any]:
        """Create laboratory interaction systems"""
        interactions = {
            'equipment_operation': {
                'instrument_controls': True,
                'parameter_setting': True,
                'method_selection': True,
                'data_collection': True
            },
            'sample_handling': {
                'pipetting': True,
                'weighing': True,
                'mixing': True,
                'heating': True
            },
            'safety_procedures': {
                'ppe_wearing': True,
                'chemical_handling': True,
                'waste_disposal': True,
                'emergency_procedures': True
            },
            'data_analysis': {
                'result_viewing': True,
                'data_export': True,
                'report_generation': True,
                'statistical_analysis': True
            }
        }

        return interactions

    def _setup_laboratory_collaboration(self, config: VREnvironment) -> Dict[str, Any]:
        """Setup collaboration features for virtual laboratory"""
        collaboration = {
            'multi_user': {
                'max_users': config.max_users,
                'user_avatars': True,
                'hand_tracking': True,
                'voice_chat': True
            },
            'shared_workspace': {
                'shared_experiments': True,
                'synchronized_equipment': True,
                'shared_data': True,
                'collaborative_notes': True
            },
            'mentoring': {
                'instructor_mode': True,
                'guided_procedures': True,
                'progress_tracking': True,
                'assessment_tools': True
            },
            'recording': {
                'session_recording': True,
                'screenshot_capture': True,
                'procedure_playback': True,
                'audit_trail': True
            }
        }

        return collaboration

    def _setup_laboratory_physics(self, config: VREnvironment) -> Dict[str, Any]:
        """Setup physics simulation for virtual laboratory"""
        physics = {
            'enabled': config.physics_enabled,
            'gravity': -9.81,  # m/s²
            'collision_detection': True,
            'fluid_simulation': {
                'enabled': True,
                'viscosity_simulation': True,
                'mixing_dynamics': True,
                'heat_transfer': True
            },
            'chemical_reactions': {
                'enabled': True,
                'reaction_kinetics': True,
                'color_changes': True,
                'gas_evolution': True
            },
            'equipment_physics': {
                'heating_simulation': True,
                'cooling_simulation': True,
                'pressure_effects': True,
                'evaporation': True
            }
        }

        return physics

    def generate_immersive_scene_description(self, viz_id: str) -> Dict[str, Any]:
        """Generate comprehensive scene description for immersive rendering"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")

        viz_data = self.visualizations[viz_id]
        viz_type = viz_data['type']

        if viz_type == 'molecular':
            return self._generate_molecular_scene_description(viz_data)
        elif viz_type == 'volumetric':
            return self._generate_volumetric_scene_description(viz_data)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

    def _generate_molecular_scene_description(self, viz_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate molecular scene description for rendering engines"""
        scene = viz_data['scene']
        config = viz_data['config']

        description = {
            'scene_type': 'molecular_visualization',
            'rendering_quality': config.rendering_quality.value,
            'objects': [],
            'lighting': {
                'ambient': {'color': [0.2, 0.2, 0.2], 'intensity': 0.3},
                'directional': [
                    {'direction': [1, -1, -1], 'color': [1, 1, 1], 'intensity': 0.7},
                    {'direction': [-1, -1, 1], 'color': [0.8, 0.8, 1], 'intensity': 0.4}
                ]
            },
            'camera': {
                'position': [0, 0, 5],
                'target': [0, 0, 0],
                'up': [0, 1, 0],
                'fov': 60
            },
            'post_processing': {
                'anti_aliasing': True,
                'ambient_occlusion': config.rendering_quality in [RenderingQuality.HIGH, RenderingQuality.ULTRA],
                'depth_of_field': False,
                'bloom': False
            }
        }

        # Add atom objects
        for atom in scene['atoms']:
            atom_object = {
                'type': 'sphere',
                'id': f"atom_{atom['metadata']['element']}",
                'position': atom['position'],
                'scale': [atom['radius']] * 3,
                'material': {
                    'type': atom['material'],
                    'color': atom['color'],
                    'metallic': 0.0,
                    'roughness': 0.3,
                    'emission': [0, 0, 0]
                },
                'metadata': atom['metadata']
            }
            description['objects'].append(atom_object)

        # Add bond objects
        for bond in scene['bonds']:
            bond_object = {
                'type': 'cylinder',
                'id': f"bond_{len(description['objects'])}",
                'start_position': bond['start_position'],
                'end_position': bond['end_position'],
                'radius': bond['radius'],
                'material': {
                    'type': bond['material'],
                    'color': bond['color'],
                    'metallic': 0.2,
                    'roughness': 0.4
                },
                'metadata': bond['metadata']
            }
            description['objects'].append(bond_object)

        return description

    def _generate_volumetric_scene_description(self, viz_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volumetric scene description for rendering engines"""
        scene = viz_data['scene']
        config = viz_data['config']
        volume_data = viz_data['volume_data']

        description = {
            'scene_type': 'volumetric_visualization',
            'rendering_quality': config.rendering_quality.value,
            'volume_renderer': {
                'data_dimensions': volume_data['dimensions'],
                'voxel_size': volume_data['voxel_size'],
                'transfer_function': scene['volume_renderer']['transfer_function'],
                'sampling_rate': scene['volume_renderer']['sampling_rate'],
                'lighting_enabled': scene['volume_renderer']['lighting_enabled']
            },
            'isosurfaces': scene['isosurfaces'],
            'slicing_planes': scene['slicing_planes'],
            'lighting': {
                'ambient': {'color': [0.1, 0.1, 0.1], 'intensity': 0.2},
                'directional': [
                    {'direction': [1, -1, -1], 'color': [1, 1, 1], 'intensity': 0.8}
                ]
            },
            'camera': {
                'position': [0, 0, 10],
                'target': [0, 0, 0],
                'up': [0, 1, 0],
                'fov': 45
            }
        }

        return description

    def export_visualization_for_platform(self, viz_id: str, platform: str, format: str = 'json') -> Union[str, bytes]:
        """Export visualization for specific AR/VR platform"""
        if viz_id not in self.visualizations:
            raise ValueError(f"Visualization {viz_id} not found")

        scene_description = self.generate_immersive_scene_description(viz_id)

        if platform == 'unity':
            return self._export_for_unity(scene_description, format)
        elif platform == 'unreal':
            return self._export_for_unreal(scene_description, format)
        elif platform == 'webxr':
            return self._export_for_webxr(scene_description, format)
        elif platform == 'oculus':
            return self._export_for_oculus(scene_description, format)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _export_for_unity(self, scene_description: Dict[str, Any], format: str) -> str:
        """Export scene for Unity engine"""
        unity_scene = {
            'scene_name': scene_description.get('scene_type', 'scientific_visualization'),
            'gameObjects': [],
            'lighting': scene_description.get('lighting', {}),
            'camera': scene_description.get('camera', {}),
            'post_processing': scene_description.get('post_processing', {})
        }

        # Convert objects to Unity GameObjects
        for obj in scene_description.get('objects', []):
            unity_object = {
                'name': obj.get('id', 'ScientificObject'),
                'transform': {
                    'position': obj.get('position', [0, 0, 0]),
                    'rotation': [0, 0, 0],
                    'scale': obj.get('scale', [1, 1, 1])
                },
                'components': []
            }

            # Add mesh renderer component
            if obj['type'] in ['sphere', 'cylinder']:
                unity_object['components'].append({
                    'type': 'MeshRenderer',
                    'material': obj.get('material', {}),
                    'mesh': obj['type']
                })

            unity_scene['gameObjects'].append(unity_object)

        if format == 'json':
            return json.dumps(unity_scene, indent=2)
        else:
            raise ValueError(f"Unsupported format for Unity: {format}")

    def _export_for_webxr(self, scene_description: Dict[str, Any], format: str) -> str:
        """Export scene for WebXR"""
        webxr_scene = {
            'aframe_version': '1.4.0',
            'scene': {
                'background': 'color: #000000',
                'vr_mode_ui': 'enabled: true',
                'embedded': True
            },
            'entities': []
        }

        # Convert objects to A-Frame entities
        for obj in scene_description.get('objects', []):
            entity = {
                'tag': 'a-entity',
                'attributes': {
                    'position': ' '.join(map(str, obj.get('position', [0, 0, 0]))),
                    'geometry': f"primitive: {obj['type']}",
                    'material': self._format_aframe_material(obj.get('material', {}))
                }
            }

            if 'metadata' in obj:
                entity['attributes']['data-metadata'] = json.dumps(obj['metadata'])

            webxr_scene['entities'].append(entity)

        # Add lighting
        webxr_scene['entities'].append({
            'tag': 'a-light',
            'attributes': {
                'type': 'ambient',
                'color': '#404040'
            }
        })

        # Add camera
        camera_pos = scene_description.get('camera', {}).get('position', [0, 1.6, 3])
        webxr_scene['entities'].append({
            'tag': 'a-camera',
            'attributes': {
                'position': ' '.join(map(str, camera_pos)),
                'wasd-controls': 'enabled: true',
                'look-controls': 'enabled: true'
            }
        })

        if format == 'json':
            return json.dumps(webxr_scene, indent=2)
        elif format == 'html':
            return self._generate_aframe_html(webxr_scene)
        else:
            raise ValueError(f"Unsupported format for WebXR: {format}")

    def _format_aframe_material(self, material: Dict[str, Any]) -> str:
        """Format material for A-Frame"""
        material_str = ""
        if 'color' in material:
            color_hex = self._rgb_to_hex(material['color'])
            material_str += f"color: {color_hex}; "

        if 'metallic' in material:
            material_str += f"metalness: {material['metallic']}; "

        if 'roughness' in material:
            material_str += f"roughness: {material['roughness']}; "

        return material_str.strip()

    def _rgb_to_hex(self, rgb: List[float]) -> str:
        """Convert RGB values to hex color"""
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _generate_aframe_html(self, webxr_scene: Dict[str, Any]) -> str:
        """Generate complete A-Frame HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Scientific Visualization</title>
    <meta name="description" content="AR/VR Scientific Visualization">
    <script src="https://aframe.io/releases/{webxr_scene['aframe_version']}/aframe.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"></script>
</head>
<body>
    <a-scene {' '.join(f'{k}="{v}"' for k, v in webxr_scene['scene'].items())}>
"""

        # Add entities
        for entity in webxr_scene['entities']:
            attrs = ' '.join(f'{k}="{v}"' for k, v in entity['attributes'].items())
            html += f"        <{entity['tag']} {attrs}></{entity['tag']}>\n"

        html += """    </a-scene>
</body>
</html>"""

        return html

    def _export_for_unreal(self, scene_description: Dict[str, Any], format: str) -> str:
        """Export scene for Unreal Engine"""
        # Simplified Unreal export
        unreal_scene = {
            'level_name': scene_description.get('scene_type', 'ScientificVisualization'),
            'actors': [],
            'lighting': scene_description.get('lighting', {}),
            'camera': scene_description.get('camera', {})
        }

        return json.dumps(unreal_scene, indent=2)

    def _export_for_oculus(self, scene_description: Dict[str, Any], format: str) -> str:
        """Export scene for Oculus platform"""
        # Simplified Oculus export (could target Unity or Unreal)
        return self._export_for_unity(scene_description, format)
```

### Integration Examples

```python
# Comprehensive AR/VR scientific applications
class ARVRScientificApplications:
    def __init__(self):
        self.arvr_expert = ARVRScientificVisualizationExpert()

    def create_protein_structure_vr_experience(self, pdb_id: str) -> str:
        """Create immersive protein structure visualization"""

        # Create molecular structure (simplified)
        protein_structure = MolecularStructure(
            atoms=[
                {'element': 'C', 'position': [0, 0, 0], 'charge': 0},
                {'element': 'N', 'position': [1.5, 0, 0], 'charge': 0},
                {'element': 'C', 'position': [2.5, 1.2, 0], 'charge': 0},
                {'element': 'O', 'position': [3.8, 1.1, 0], 'charge': 0},
                {'element': 'S', 'position': [2.0, 2.5, 0], 'charge': 0}
            ],
            bonds=[(0, 1, 1), (1, 2, 1), (2, 3, 2), (2, 4, 1)],
            properties={
                'molecular_weight': 121.15,
                'charge': 0,
                'dipole_moment': 1.2
            },
            metadata={'pdb_id': pdb_id, 'source': 'protein_databank'}
        )

        # Configure visualization
        viz_config = ScientificVisualization(
            viz_id=f"protein_{pdb_id}_vr",
            viz_type=VisualizationType.MOLECULAR,
            title=f"Protein Structure: {pdb_id}",
            description="Interactive VR protein structure visualization",
            data_source=f"PDB:{pdb_id}",
            rendering_quality=RenderingQuality.HIGH,
            interaction_modes=[
                InteractionMode.HAND_TRACKING,
                InteractionMode.CONTROLLER,
                InteractionMode.VOICE
            ],
            collaboration_enabled=True,
            real_time_updates=False
        )

        # Create visualization
        viz_id = self.arvr_expert.create_molecular_visualization(protein_structure, viz_config)

        return viz_id

    def create_medical_imaging_ar_assistant(self, imaging_data: np.ndarray) -> str:
        """Create AR assistant for medical imaging analysis"""

        # Create volumetric data
        volume_data = VolumetricData(
            data=imaging_data,
            dimensions=imaging_data.shape,
            voxel_size=(1.0, 1.0, 1.0),  # mm
            origin=(0, 0, 0),
            units='HU',  # Hounsfield Units
            metadata={
                'modality': 'CT',
                'patient_id': 'anonymous',
                'scan_date': '2024-01-15'
            }
        )

        # Configure AR visualization
        viz_config = ScientificVisualization(
            viz_id="medical_imaging_ar",
            viz_type=VisualizationType.VOLUMETRIC,
            title="Medical Imaging AR Assistant",
            description="Augmented reality medical imaging analysis",
            data_source="CT_scan",
            rendering_quality=RenderingQuality.MEDIUM,
            interaction_modes=[InteractionMode.GESTURE, InteractionMode.VOICE],
            collaboration_enabled=True,
            real_time_updates=True
        )

        # Create visualization
        viz_id = self.arvr_expert.create_volumetric_visualization(volume_data, viz_config)

        return viz_id

    def create_chemistry_virtual_lab(self) -> str:
        """Create virtual chemistry laboratory"""

        # Configure virtual laboratory
        lab_config = VREnvironment(
            environment_id="chemistry_lab_vr",
            name="Virtual Chemistry Laboratory",
            description="Immersive chemistry laboratory for education and research",
            scene_type="chemistry_laboratory",
            lighting_setup={
                'type': 'laboratory_fluorescent',
                'intensity': 1.0,
                'color_temperature': 4000,
                'shadows': True
            },
            physics_enabled=True,
            collaborative=True,
            max_users=8,
            background_sound="laboratory_ambience"
        )

        # Create laboratory
        lab_id = self.arvr_expert.create_virtual_laboratory(lab_config)

        return lab_id

    def create_astronomical_data_vr_exploration(self, star_catalog: pd.DataFrame) -> str:
        """Create VR exploration of astronomical data"""

        # Process astronomical data
        star_positions = []
        star_properties = []

        for _, star in star_catalog.iterrows():
            # Convert celestial coordinates to 3D positions
            ra = star.get('ra', 0)  # Right ascension
            dec = star.get('dec', 0)  # Declination
            distance = star.get('distance', 100)  # parsecs

            # Convert to Cartesian coordinates
            x = distance * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
            y = distance * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
            z = distance * np.sin(np.radians(dec))

            star_positions.append([x, y, z])
            star_properties.append({
                'magnitude': star.get('magnitude', 5),
                'spectral_class': star.get('spectral_class', 'G'),
                'name': star.get('name', f'Star_{len(star_positions)}')
            })

        # Create star field visualization
        star_field_data = {
            'positions': star_positions,
            'properties': star_properties,
            'scale_factor': 0.1  # Scale down for VR
        }

        # Configure visualization
        viz_config = ScientificVisualization(
            viz_id="astronomical_vr_exploration",
            viz_type=VisualizationType.SPATIAL,
            title="Astronomical Data VR Exploration",
            description="Immersive exploration of star catalog data",
            data_source="hipparcos_catalog",
            rendering_quality=RenderingQuality.HIGH,
            interaction_modes=[
                InteractionMode.CONTROLLER,
                InteractionMode.GAZE,
                InteractionMode.VOICE
            ],
            collaboration_enabled=True,
            real_time_updates=False
        )

        # Store visualization (simplified - would create full 3D scene)
        self.arvr_expert.visualizations[viz_config.viz_id] = {
            'type': 'astronomical',
            'data': star_field_data,
            'config': viz_config,
            'created_at': datetime.now()
        }

        return viz_config.viz_id

    def create_climate_data_immersive_analysis(self, climate_data: np.ndarray) -> str:
        """Create immersive climate data analysis environment"""

        # Create climate data visualization
        volume_data = VolumetricData(
            data=climate_data,
            dimensions=climate_data.shape,
            voxel_size=(1.0, 1.0, 1.0),  # degrees
            origin=(-180, -90, 0),  # longitude, latitude, time
            units='temperature_celsius',
            metadata={
                'variable': 'surface_temperature',
                'time_range': '2020-2024',
                'resolution': '1_degree'
            }
        )

        # Configure immersive visualization
        viz_config = ScientificVisualization(
            viz_id="climate_data_immersive",
            viz_type=VisualizationType.TEMPORAL,
            title="Climate Data Immersive Analysis",
            description="4D climate data exploration in VR",
            data_source="climate_model_output",
            rendering_quality=RenderingQuality.HIGH,
            interaction_modes=[
                InteractionMode.HAND_TRACKING,
                InteractionMode.CONTROLLER
            ],
            collaboration_enabled=True,
            real_time_updates=True
        )

        # Create visualization
        viz_id = self.arvr_expert.create_volumetric_visualization(volume_data, viz_config)

        return viz_id

    def generate_webxr_molecular_viewer(self, viz_id: str) -> str:
        """Generate WebXR molecular viewer"""
        return self.arvr_expert.export_visualization_for_platform(viz_id, 'webxr', 'html')

    def create_collaborative_research_session(self, viz_ids: List[str]) -> Dict[str, Any]:
        """Create collaborative VR research session"""

        session_id = f"collab_session_{int(time.time())}"

        session_config = {
            'session_id': session_id,
            'title': "Collaborative Scientific Analysis",
            'visualizations': viz_ids,
            'max_participants': 6,
            'session_type': 'research_collaboration',
            'features': {
                'voice_chat': True,
                'shared_annotations': True,
                'synchronized_navigation': True,
                'recording': True,
                'whiteboard': True
            },
            'created_at': datetime.now()
        }

        self.arvr_expert.collaboration_sessions[session_id] = session_config

        return session_config
```

## Use Cases

### Molecular & Chemical Visualization
- **Drug Discovery**: Interactive protein-drug binding visualization, molecular dynamics exploration
- **Chemical Education**: 3D molecular orbital visualization, reaction mechanism animation
- **Materials Science**: Crystal structure analysis, defect visualization, property mapping
- **Biochemistry**: Enzyme mechanism studies, metabolic pathway exploration

### Medical & Biological Applications
- **Medical Training**: Virtual anatomy lessons, surgical procedure simulation
- **Diagnostic Assistance**: AR-enhanced medical imaging, real-time data overlay
- **Research Collaboration**: Multi-user virtual microscopy, collaborative data analysis
- **Patient Education**: Interactive disease progression visualization, treatment explanation

### Environmental & Earth Sciences
- **Climate Visualization**: 4D climate data exploration, global warming simulation
- **Geological Survey**: Virtual field trips, rock formation analysis, seismic data visualization
- **Oceanography**: Underwater environment simulation, current flow visualization
- **Atmospheric Science**: Weather pattern analysis, pollution dispersion modeling

### Physics & Engineering
- **Particle Physics**: Event visualization, detector simulation, collision analysis
- **Fluid Dynamics**: Flow field visualization, turbulence analysis, aerodynamic studies
- **Electromagnetics**: Field visualization, antenna pattern analysis, wave propagation
- **Quantum Mechanics**: Wave function visualization, quantum state evolution

### Educational Applications
- **Virtual Laboratories**: Safe experimentation environments, equipment training
- **Concept Visualization**: Abstract concept illustration, interactive learning
- **Remote Learning**: Distributed virtual classrooms, hands-on experiences
- **Assessment Tools**: Interactive testing, practical skill evaluation

## Integration with Existing Agents

- **Visualization Expert**: Enhanced 2D/3D plotting with immersive capabilities
- **Digital Twins Expert**: AR/VR interfaces for digital twin interaction
- **Molecular Simulation Expert**: Immersive molecular dynamics visualization
- **Neural Networks Expert**: 3D neural network architecture visualization
- **Scientific Database Expert**: Immersive data exploration and analysis
- **Experiment Manager**: Virtual experiment design and execution

This agent transforms flat scientific data into immersive, interactive experiences that enhance understanding, collaboration, and discovery through cutting-edge AR/VR technologies.