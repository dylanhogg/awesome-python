// NOTE: this recreates a hosted zone with desc "Managed by Terraform" next to existing
// manually created infocruncher.com zone:
//resource "aws_route53_zone" "route53_zone" {
//  name = var.base_url
//  tags = var.common_tags
//}

resource "aws_route53_record" "r53_record" {
  // zone_id = aws_route53_zone.route53_zone.zone_id
  zone_id = var.existing_hosted_zone_id  // Non-TF managed infocruncher.com hosted zone ID.
  name    = var.domain
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.www_s3_distribution.domain_name
    zone_id                = aws_cloudfront_distribution.www_s3_distribution.hosted_zone_id
    evaluate_target_health = false
  }
}

# Add R53 records for certificate DNS validation
# Ref: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate#referencing-domain_validation_options-with-for_each-based-resources
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.ssl_certificate.domain_validation_options : dvo.domain_name => {
      name    = dvo.resource_record_name
      record  = dvo.resource_record_value
      type    = dvo.resource_record_type
      // zone_id = aws_route53_zone.route53_zone.zone_id
      zone_id = var.existing_hosted_zone_id  // Non-TF managed infocruncher.com hosted zone ID.
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 300
  type            = each.value.type
  zone_id         = each.value.zone_id
}