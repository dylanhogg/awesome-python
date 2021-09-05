resource "aws_s3_bucket" "s3_bucket" {
  // Bucket name must be unique and must not contain spaces, uppercase letters or underscores.
  bucket = var.domain
  acl    = "public-read"

  versioning {
    enabled = true
  }

  website {
    index_document = var.index_document
    error_document = var.error_document
  }

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }

  tags = var.common_tags
}

resource "aws_s3_bucket_policy" "s3_bucket_policy" {
  bucket = aws_s3_bucket.s3_bucket.id

  policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
      {
          "Sid": "PublicReadGetObject",
          "Effect": "Allow",
          "Principal": "*",
          "Action": [
             "s3:GetObject"
          ],
          "Resource": [
             "arn:aws:s3:::${aws_s3_bucket.s3_bucket.id}/*"
          ]
      }
    ]
}
POLICY
}
