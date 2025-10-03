"""
Unit Tests for New Agents (2025-10-02 Enhancement)

Tests for:
- Hardness Testing Agent
- Thermal Conductivity Agent
- Corrosion Agent
- X-ray Microscopy Agent
- Monte Carlo Agent

Author: Materials Characterization Agents Team
Version: 1.0.0
Date: 2025-10-02
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import new agents
from mechanical.hardness_testing_agent import HardnessTestingAgent
from thermal.thermal_conductivity_agent import ThermalConductivityAgent
from electrochemical.corrosion_agent import CorrosionAgent
from microscopy.xray_microscopy_agent import XRayMicroscopyAgent
from computational.monte_carlo_agent import MonteCarloAgent


class TestHardnessTestingAgent(unittest.TestCase):
    """Test cases for Hardness Testing Agent."""

    def setUp(self):
        """Initialize agent for testing."""
        self.agent = HardnessTestingAgent()

    def test_vickers_hardness(self):
        """Test Vickers hardness measurement."""
        result = self.agent.execute({
            'technique': 'vickers',
            'load_n': 9.807,  # 1 kgf
            'diagonal_measurements_um': [42.5, 43.1, 42.8, 43.0, 42.6]
        })

        self.assertEqual(result['technique'], 'vickers_microhardness')
        self.assertIn('hardness_value', result)
        self.assertIn('hardness_std', result)
        self.assertIn('estimated_tensile_strength_mpa', result)
        self.assertGreater(result['hardness_value'], 0)
        self.assertEqual(result['number_of_indents'], 5)

    def test_rockwell_hardness(self):
        """Test Rockwell hardness measurement."""
        result = self.agent.execute({
            'technique': 'rockwell',
            'scale': 'HRC',
            'measurements': [58.5, 59.0, 58.8, 59.2, 58.7]
        })

        self.assertEqual(result['technique'], 'rockwell_hardness')
        self.assertEqual(result['unit'], 'HRC')
        self.assertIn('hardness_value', result)
        self.assertIn('equivalent_vickers_hv', result)
        self.assertTrue(result['in_valid_range'])

    def test_shore_hardness(self):
        """Test Shore durometer for polymers."""
        result = self.agent.execute({
            'technique': 'shore',
            'scale': 'Shore_A',
            'measurements': [62, 64, 63, 62, 63]
        })

        self.assertEqual(result['technique'], 'shore_durometer')
        self.assertEqual(result['unit'], 'Shore_A')
        self.assertIn('estimated_tensile_modulus_mpa', result)
        self.assertGreater(result['hardness_value'], 0)
        self.assertLess(result['hardness_value'], 100)

    def test_hardness_scale_conversion(self):
        """Test conversion between hardness scales."""
        result = self.agent.execute({
            'technique': 'conversion',
            'source_scale': 'HV',
            'source_value': 500,
            'target_scales': ['HRC', 'HB']
        })

        self.assertEqual(result['technique'], 'hardness_scale_conversion')
        self.assertIn('conversions', result)
        self.assertIn('HRC', result['conversions'])
        self.assertIn('HB', result['conversions'])

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        capabilities = self.agent.get_capabilities()

        self.assertEqual(capabilities['agent_type'], 'hardness_testing')
        self.assertIn('vickers', capabilities['supported_techniques'])
        self.assertIn('rockwell', capabilities['supported_techniques'])


class TestThermalConductivityAgent(unittest.TestCase):
    """Test cases for Thermal Conductivity Agent."""

    def setUp(self):
        """Initialize agent for testing."""
        self.agent = ThermalConductivityAgent()

    def test_laser_flash_analysis(self):
        """Test Laser Flash Analysis."""
        result = self.agent.execute({
            'technique': 'laser_flash',
            'thickness_mm': 2.0,
            'density_g_cm3': 2.60,
            'specific_heat_j_g_k': 0.808,
            'temperature_k': 298,
            'time_to_half_max_ms': 500
        })

        self.assertEqual(result['technique'], 'laser_flash_analysis')
        self.assertIn('thermal_diffusivity_mm2_s', result)
        self.assertIn('thermal_conductivity_w_m_k', result)
        self.assertGreater(result['thermal_diffusivity_mm2_s'], 0)
        self.assertGreater(result['thermal_conductivity_w_m_k'], 0)

    def test_hot_disk(self):
        """Test Hot Disk (TPS) measurement."""
        result = self.agent.execute({
            'technique': 'hot_disk',
            'expected_conductivity_w_m_k': 0.2,
            'expected_diffusivity_mm2_s': 0.15,
            'sensor_power_w': 0.03,
            'measurement_time_s': 20
        })

        self.assertEqual(result['technique'], 'hot_disk_tps')
        self.assertIn('thermal_conductivity_w_m_k', result)
        self.assertIn('thermal_diffusivity_mm2_s', result)
        self.assertIn('volumetric_heat_capacity_mj_m3_k', result)

    def test_anisotropic_conductivity(self):
        """Test anisotropic thermal conductivity measurement."""
        result = self.agent.execute({
            'technique': 'anisotropy',
            'k_in_plane_w_m_k': 150,
            'k_through_plane_w_m_k': 5
        })

        self.assertEqual(result['technique'], 'anisotropic_thermal_conductivity')
        self.assertIn('anisotropy_ratio', result)
        self.assertEqual(result['anisotropy_ratio'], 30.0)
        self.assertIn('anisotropy_classification', result)

    def test_temperature_sweep(self):
        """Test temperature-dependent conductivity."""
        result = self.agent.execute({
            'technique': 'temperature_sweep',
            'base_technique': 'laser_flash',
            'temperature_range_k': (200, 800),
            'num_points': 11
        })

        self.assertEqual(result['technique'], 'temperature_dependent_conductivity')
        self.assertIn('temperatures_k', result)
        self.assertIn('thermal_conductivity_w_m_k', result)
        self.assertEqual(len(result['temperatures_k']), 11)

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        capabilities = self.agent.get_capabilities()

        self.assertEqual(capabilities['agent_type'], 'thermal_conductivity')
        self.assertIn('laser_flash', capabilities['supported_techniques'])
        self.assertIn('hot_disk', capabilities['supported_techniques'])


class TestCorrosionAgent(unittest.TestCase):
    """Test cases for Corrosion Agent."""

    def setUp(self):
        """Initialize agent for testing."""
        self.agent = CorrosionAgent()

    def test_potentiodynamic_polarization(self):
        """Test Tafel polarization."""
        result = self.agent.execute({
            'technique': 'potentiodynamic_polarization',
            'material': 'carbon_steel',
            'electrolyte': '3.5% NaCl',
            'temperature_c': 25
        })

        self.assertEqual(result['technique'], 'potentiodynamic_polarization')
        self.assertIn('corrosion_potential_v_vs_ref', result)
        self.assertIn('corrosion_current_density_a_cm2', result)
        self.assertIn('corrosion_rate_mm_per_year', result)
        self.assertIn('anodic_tafel_slope_v_decade', result)
        self.assertIn('cathodic_tafel_slope_v_decade', result)

    def test_linear_polarization_resistance(self):
        """Test LPR measurement."""
        result = self.agent.execute({
            'technique': 'linear_polarization_resistance',
            'material': 'stainless_steel_304',
            'electrolyte': '3.5% NaCl',
            'polarization_resistance_ohm_cm2': 50000
        })

        self.assertEqual(result['technique'], 'linear_polarization_resistance')
        self.assertIn('polarization_resistance_ohm_cm2', result)
        self.assertIn('corrosion_current_density_a_cm2', result)
        self.assertIn('stern_geary_constant_v', result)
        self.assertEqual(result['polarization_resistance_ohm_cm2'], 50000)

    def test_cyclic_polarization(self):
        """Test cyclic polarization for pitting."""
        result = self.agent.execute({
            'technique': 'cyclic_polarization',
            'material': 'stainless_steel_316L',
            'electrolyte': '3.5% NaCl',
            'e_pit_v': 0.350,
            'e_prot_v': 0.150
        })

        self.assertEqual(result['technique'], 'cyclic_polarization')
        self.assertIn('pitting_potential_v', result)
        self.assertIn('protection_potential_v', result)
        self.assertIn('pitting_susceptibility', result)

    def test_salt_spray(self):
        """Test salt spray testing."""
        result = self.agent.execute({
            'technique': 'salt_spray',
            'material': 'zinc_coated_steel',
            'coating_thickness_um': 25,
            'exposure_hours': 96,
            'percent_area_corroded': 3
        })

        self.assertEqual(result['technique'], 'salt_spray_test_astm_b117')
        self.assertIn('corrosion_rating', result)
        self.assertIn('passes_requirement', result)

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        capabilities = self.agent.get_capabilities()

        self.assertEqual(capabilities['agent_type'], 'corrosion')
        self.assertIn('potentiodynamic_polarization', capabilities['supported_techniques'])
        self.assertIn('linear_polarization_resistance', capabilities['supported_techniques'])


class TestXRayMicroscopyAgent(unittest.TestCase):
    """Test cases for X-ray Microscopy Agent."""

    def setUp(self):
        """Initialize agent for testing."""
        self.agent = XRayMicroscopyAgent()

    def test_xray_computed_tomography(self):
        """Test X-ray CT."""
        result = self.agent.execute({
            'technique': 'xray_computed_tomography',
            'sample_name': 'metal_foam',
            'photon_energy_kev': 25,
            'num_projections': 1800,
            'detector_pixel_size_um': 6.5,
            'measured_porosity_percent': 35.0
        })

        self.assertEqual(result['technique'], 'xray_computed_tomography')
        self.assertIn('voxel_size_um', result)
        self.assertIn('reconstructed_volume', result)
        self.assertIn('analysis_results', result)
        self.assertEqual(result['analysis_results']['porosity_percent'], 35.0)

    def test_xray_fluorescence_microscopy(self):
        """Test XFM elemental mapping."""
        result = self.agent.execute({
            'technique': 'xray_fluorescence_microscopy',
            'sample_name': 'catalyst_particle',
            'incident_energy_kev': 12,
            'beam_size_um': 0.5,
            'elements_detected': ['Pt', 'Ru', 'C'],
            'scan_area_um': (50, 50)
        })

        self.assertEqual(result['technique'], 'xray_fluorescence_microscopy')
        self.assertIn('spatial_resolution_um', result)
        self.assertIn('elemental_data', result)
        self.assertEqual(result['elements_detected'], ['Pt', 'Ru', 'C'])

    def test_ptychography(self):
        """Test ptychography reconstruction."""
        result = self.agent.execute({
            'technique': 'ptychography',
            'sample_name': 'nanostructure',
            'photon_energy_kev': 6.2,
            'achieved_resolution_nm': 8,
            'scan_points': 144
        })

        self.assertEqual(result['technique'], 'xray_ptychography')
        self.assertIn('achieved_resolution_nm', result)
        self.assertIn('reconstruction', result)
        self.assertLess(result['achieved_resolution_nm'], 10)

    def test_stxm(self):
        """Test Scanning TXM."""
        result = self.agent.execute({
            'technique': 'scanning_txm',
            'sample_name': 'carbon_nanoparticles',
            'scan_range_um': (10, 10),
            'step_size_nm': 30,
            'perform_xanes': True
        })

        self.assertEqual(result['technique'], 'scanning_transmission_xray_microscopy')
        self.assertIn('xanes_enabled', result)
        self.assertTrue(result['xanes_enabled'])
        self.assertIn('image_pixels', result)

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        capabilities = self.agent.get_capabilities()

        self.assertEqual(capabilities['agent_type'], 'xray_microscopy')
        self.assertIn('transmission_xray_microscopy', capabilities['supported_techniques'])
        self.assertIn('xray_computed_tomography', capabilities['supported_techniques'])


class TestMonteCarloAgent(unittest.TestCase):
    """Test cases for Monte Carlo Agent."""

    def setUp(self):
        """Initialize agent for testing."""
        self.agent = MonteCarloAgent()

    def test_metropolis_mc(self):
        """Test Metropolis Monte Carlo."""
        result = self.agent.execute({
            'technique': 'metropolis',
            'temperature_k': 298,
            'num_particles': 256,
            'box_length_nm': 3.0,
            'num_production_steps': 10000
        })

        self.assertEqual(result['technique'], 'metropolis_monte_carlo')
        self.assertEqual(result['ensemble'], 'NVT (canonical)')
        self.assertIn('acceptance_rate_production', result)
        self.assertIn('thermodynamic_averages', result)
        self.assertGreater(result['acceptance_rate_production'], 0)
        self.assertLess(result['acceptance_rate_production'], 1)

    def test_grand_canonical_mc(self):
        """Test GCMC."""
        result = self.agent.execute({
            'technique': 'grand_canonical',
            'temperature_k': 300,
            'chemical_potential_kj_mol': -15,
            'box_length_nm': 3.0,
            'num_steps': 10000
        })

        self.assertEqual(result['technique'], 'grand_canonical_monte_carlo')
        self.assertEqual(result['ensemble'], 'μVT (grand canonical)')
        self.assertIn('average_num_particles', result)
        self.assertIn('acceptance_rates', result)

    def test_kinetic_monte_carlo(self):
        """Test KMC."""
        result = self.agent.execute({
            'technique': 'kinetic_monte_carlo',
            'lattice_size': 50,
            'rate_adsorption_per_s': 1e6,
            'rate_desorption_per_s': 1e3,
            'num_events': 1000
        })

        self.assertEqual(result['technique'], 'kinetic_monte_carlo')
        self.assertIn('total_time_s', result)
        self.assertIn('final_coverage', result)
        self.assertGreater(result['total_time_s'], 0)

    def test_wang_landau(self):
        """Test Wang-Landau sampling."""
        result = self.agent.execute({
            'technique': 'wang_landau',
            'energy_min_kj_mol': -100,
            'energy_max_kj_mol': 0,
            'energy_bins': 50,
            'f_final': 1e-6
        })

        self.assertEqual(result['technique'], 'wang_landau_sampling')
        self.assertIn('log_density_of_states', result)
        self.assertIn('free_energy_kj_mol', result)
        self.assertEqual(result['num_energy_bins'], 50)

    def test_parallel_tempering(self):
        """Test parallel tempering."""
        result = self.agent.execute({
            'technique': 'parallel_tempering',
            'temp_min_k': 250,
            'temp_max_k': 500,
            'num_replicas': 8,
            'num_steps': 10000
        })

        self.assertEqual(result['technique'], 'parallel_tempering_monte_carlo')
        self.assertIn('exchange_acceptance_rate', result)
        self.assertIn('average_energies_kj_mol', result)
        self.assertEqual(result['num_replicas'], 8)

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        capabilities = self.agent.get_capabilities()

        self.assertEqual(capabilities['agent_type'], 'monte_carlo')
        self.assertIn('metropolis', capabilities['supported_techniques'])
        self.assertIn('grand_canonical', capabilities['supported_techniques'])


class TestCrossValidations(unittest.TestCase):
    """Test cross-validation between new agents."""

    def test_hardness_vickers_rockwell_correlation(self):
        """Test Vickers-Rockwell hardness correlation."""
        hardness_agent = HardnessTestingAgent()

        # Vickers test
        vickers_result = hardness_agent.execute({
            'technique': 'vickers',
            'load_n': 9.807,
            'diagonal_measurements_um': [40, 41, 40.5, 40.2, 40.8]
        })

        # Rockwell test
        rockwell_result = hardness_agent.execute({
            'technique': 'rockwell',
            'scale': 'HRC',
            'measurements': [60, 61, 60.5, 60.2, 60.8]
        })

        # Both should report hardness
        self.assertIn('hardness_value', vickers_result)
        self.assertIn('hardness_value', rockwell_result)

    def test_lfa_hot_disk_correlation(self):
        """Test LFA vs Hot Disk thermal conductivity."""
        thermal_agent = ThermalConductivityAgent()

        # LFA
        lfa_result = thermal_agent.execute({
            'technique': 'laser_flash',
            'thickness_mm': 2.0,
            'density_g_cm3': 2.0,
            'specific_heat_j_g_k': 0.8,
            'temperature_k': 298
        })

        # Hot Disk
        hot_disk_result = thermal_agent.execute({
            'technique': 'hot_disk',
            'expected_conductivity_w_m_k': lfa_result['thermal_conductivity_w_m_k'],
            'measurement_time_s': 20
        })

        # Both should report thermal conductivity
        self.assertIn('thermal_conductivity_w_m_k', lfa_result)
        self.assertIn('thermal_conductivity_w_m_k', hot_disk_result)

    def test_tafel_lpr_corrosion_correlation(self):
        """Test Tafel vs LPR corrosion rate."""
        corrosion_agent = CorrosionAgent()

        # Tafel
        tafel_result = corrosion_agent.execute({
            'technique': 'potentiodynamic_polarization',
            'material': 'steel',
            'electrolyte': '3.5% NaCl'
        })

        # LPR
        lpr_result = corrosion_agent.execute({
            'technique': 'linear_polarization_resistance',
            'material': 'steel',
            'electrolyte': '3.5% NaCl',
            'beta_a': tafel_result['anodic_tafel_slope_v_decade'],
            'beta_c': tafel_result['cathodic_tafel_slope_v_decade']
        })

        # Both should report corrosion rate
        self.assertIn('corrosion_rate_mm_per_year', tafel_result)
        self.assertIn('corrosion_rate_mm_per_year', lpr_result)


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHardnessTestingAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalConductivityAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestCorrosionAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestXRayMicroscopyAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMonteCarloAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossValidations))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
