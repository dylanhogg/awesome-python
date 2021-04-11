resource "aws_route53_record" "www" {
  // zone_id = aws_route53_zone.route53_zone.zone_id
  zone_id = var.existing_hosted_zone_id  // Non-TF managed infocruncher.com hosted zone ID.
  name    = var.domain
  type    = "A"

  alias {
    name                   = aws_s3_bucket.s3_bucket.website_domain
    zone_id                = aws_s3_bucket.s3_bucket.hosted_zone_id
    evaluate_target_health = true
  }

  depends_on = [
    aws_s3_bucket.s3_bucket
  ]
}

// NOTE: this recreates a hosted zone with desc "Managed by Terraform" next to existing
// manually created infocruncher.com zone:
//resource "aws_route53_zone" "route53_zone" {
//  name = var.base_url
//}
