aws_profile="prd-non-tf-905234897161"
region="us-east-1"
env="prd"
app_name="crazy-awesome-python"
domain="awesome-python.infocruncher.com"  # Also bucket name
base_url="infocruncher.com"
existing_hosted_zone_id="Z22XOKU6RYOC4M"
ttl=86400
index_document="app.html"
error_document="error.html"
notfound_document="404.html"
common_tags = {
  tag_version = "1.0"
  deployment  = "tf"
  app_name    = "crazy-awesome-python"
  env         = "prd"
}
