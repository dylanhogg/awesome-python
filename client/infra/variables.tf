variable "aws_profile" {
  type=string
}

variable "region" {
  type=string
}

variable "env" {
  type=string
}

variable "app_name" {
  type=string
}

variable "domain" {
  type=string
}

variable "base_url" {
  type=string
}

variable "existing_hosted_zone_id" {
  type=string
}

variable "index_document" {
  type=string
}

variable "error_document" {
  type=string
}

variable "notfound_document" {
  type=string
}

variable "ttl" {
  type=number
}

variable "common_tags" { }
