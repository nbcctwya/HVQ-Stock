import unittest

import torch

from trainer.train_ypred import ReturnPredictor


class GatedFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.P, self.K = 8, 13, 128
        self.alpha = torch.randn(self.B)
        self.beta_p = torch.randn(self.B, self.P)
        self.beta_l = torch.randn(self.B, self.K)
        self.f_prior = torch.randn(self.B, self.P)
        self.f_latent = torch.randn(self.B, self.K)
        self.prior_term = (self.beta_p * self.f_prior).sum(dim=1)
        self.latent_term = (self.beta_l * self.f_latent).sum(dim=1)

    def _forward(self, predictor):
        return predictor(
            alpha=self.alpha,
            beta_p=self.beta_p,
            beta_l=self.beta_l,
            f_prior=self.f_prior,
            f_latent=self.f_latent,
        )

    def test_fixed_mode_matches_original_formula(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=True, fusion='fixed')
        out = self._forward(predictor)
        expected = self.alpha + self.prior_term + self.latent_term
        self.assertEqual(out.shape, (self.B,))
        self.assertTrue(torch.allclose(out, expected))

    def test_fixed_mode_without_prior(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=False, fusion='fixed')
        out = self._forward(predictor)
        expected = self.alpha + self.latent_term
        self.assertTrue(torch.allclose(out, expected))

    def test_gated_mode_defaults_to_equal_weighting(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=True, fusion='gated')
        with torch.no_grad():
            predictor.gate.weight.zero_()
            predictor.gate.bias.zero_()
        out = self._forward(predictor)
        expected = self.alpha + 0.5 * self.prior_term + 0.5 * self.latent_term
        self.assertEqual(out.shape, (self.B,))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_gated_mode_interpolates_between_prior_and_latent(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=True, fusion='gated')
        with torch.no_grad():
            predictor.gate.weight.zero_()
            predictor.gate.bias.fill_(20.0)
        out = self._forward(predictor)
        self.assertTrue(torch.allclose(out, self.alpha + self.prior_term, atol=1e-4))
        with torch.no_grad():
            predictor.gate.bias.fill_(-20.0)
        out = self._forward(predictor)
        self.assertTrue(torch.allclose(out, self.alpha + self.latent_term, atol=1e-4))

    def test_gated_mode_gate_is_sample_adaptive(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=True, fusion='gated')
        with torch.no_grad():
            predictor.gate.bias.zero_()
        out_a = predictor(self.alpha, self.beta_p, self.beta_l,
                          self.f_prior, self.f_latent)
        out_b = predictor(self.alpha, self.beta_p, self.beta_l,
                          -self.f_prior, self.f_latent)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_gated_mode_gradient_reaches_gate(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=True, fusion='gated')
        out = self._forward(predictor)
        out.sum().backward()
        self.assertIsNotNone(predictor.gate.weight.grad)
        self.assertTrue(torch.isfinite(predictor.gate.weight.grad).all())
        self.assertGreater(predictor.gate.weight.grad.abs().sum().item(), 0.0)

    def test_use_prior_false_ignores_gate(self):
        predictor = ReturnPredictor(self.P, self.K, use_prior=False, fusion='gated')
        out = self._forward(predictor)
        expected = self.alpha + self.latent_term
        self.assertTrue(torch.allclose(out, expected))

    def test_unknown_fusion_mode_raises(self):
        with self.assertRaises(AssertionError):
            ReturnPredictor(self.P, self.K, fusion='concat')


if __name__ == '__main__':
    unittest.main()
