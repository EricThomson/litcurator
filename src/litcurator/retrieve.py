"""
PubMed retrieval via NCBI E-utilities.

Fetches articles from a configured list of journals for a given date range,
returning results as a list of dicts saved to JSON.

Pipeline:
    build_query -> search_pubmed -> fetch_articles (-> parse_articles_xml -> parse_single_article)
    All orchestrated by retrieve_range(), the public entry point.
"""

import calendar
import json
import os
import xml.etree.ElementTree as ET
from datetime import date

import requests
from dotenv import load_dotenv

from litcurator import db
from litcurator.config import JOURNALS

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def build_query(journal, neuro_keyword, start_date, end_date):
    """Build the PubMed search query string for a single journal and date range."""
    date_range = (
        f'("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : '
        f'"{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
    )
    journal_filter = f'"{journal}"[Journal]'
    abstract_filter = "hasabstract"

    if neuro_keyword:
        query = f"{date_range} AND {journal_filter} AND neuroscience AND {abstract_filter}"
    else:
        query = f"{date_range} AND {journal_filter} AND {abstract_filter}"

    return query


def search_pubmed(query):
    """Run esearch and return a list of PMIDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 200,
        "retmode": "json",
        "api_key": NCBI_API_KEY,
    }
    response = requests.get(ESEARCH_URL, params=params)
    response.raise_for_status()

    data = response.json()
    pmids = data["esearchresult"]["idlist"]
    return pmids


def fetch_articles(pmids):
    """Run efetch for a list of PMIDs and return parsed article dicts."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "api_key": NCBI_API_KEY,
    }
    response = requests.get(EFETCH_URL, params=params)
    response.raise_for_status()

    return parse_articles_xml(response.text)


def parse_articles_xml(xml_text):
    """Parse PubMed XML response into a list of article dicts."""
    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        parsed = parse_single_article(article)
        articles.append(parsed)

    return articles


def parse_single_article(article):
    """Extract fields from a single PubmedArticle XML element."""
    # Title
    title_el = article.find(".//ArticleTitle")
    title = "".join(title_el.itertext()) if title_el is not None else ""

    # Abstract — may have multiple AbstractText elements (e.g. structured abstracts).
    # Use itertext() to capture text inside nested tags like <sup>, <sub>, <i>, etc.
    abstract_parts = article.findall(".//AbstractText")
    abstract = " ".join("".join(part.itertext()) for part in abstract_parts)

    # Journal name
    journal_el = article.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None else ""

    # Publication date — fall back to MedlineDate if structured date is missing
    year_el = article.find(".//PubDate/Year")
    month_el = article.find(".//PubDate/Month")
    pub_date = ""
    if year_el is not None:
        pub_date = year_el.text
        if month_el is not None:
            pub_date = f"{year_el.text}-{month_el.text}"
    else:
        medline_el = article.find(".//PubDate/MedlineDate")
        if medline_el is not None:
            pub_date = medline_el.text

    # Electronic publication date (epub ahead of print)
    epub_date = ""
    for article_date in article.findall(".//ArticleDate"):
        if article_date.get("DateType") == "Electronic":
            epub_year = article_date.findtext("Year", default="")
            epub_month = article_date.findtext("Month", default="")
            epub_day = article_date.findtext("Day", default="")
            if epub_year and epub_month and epub_day:
                epub_date = f"{epub_year}-{epub_month}-{epub_day}"
            break

    # Electronic publication date (epub ahead of print)
    epub_date = ""
    for article_date in article.findall(".//ArticleDate"):
        if article_date.get("DateType") == "Electronic":
            epub_year = article_date.findtext("Year", default="")
            epub_month = article_date.findtext("Month", default="")
            epub_day = article_date.findtext("Day", default="")
            if epub_year and epub_month and epub_day:
                epub_date = f"{epub_year}-{epub_month}-{epub_day}"
            break

    # Authors and affiliations
    author_els = article.findall(".//Author")
    authors = []
    for author in author_els:
        last = author.findtext("LastName", default="")
        fore = author.findtext("ForeName", default="")
        full_name = f"{fore} {last}".strip()
        affiliation_el = author.find(".//AffiliationInfo/Affiliation")
        affiliation = affiliation_el.text if affiliation_el is not None else ""
        if full_name:
            authors.append({"name": full_name, "affiliation": affiliation})

    # Publication types (e.g. "Journal Article", "Review", "Comment", "News")
    pub_type_els = article.findall(".//PublicationType")
    pub_types = [el.text for el in pub_type_els if el.text]

    # PubMed ID
    pmid_el = article.find(".//PMID")
    pubmed_id = pmid_el.text if pmid_el is not None else ""

    # DOI
    doi = ""
    for article_id in article.findall(".//ArticleId"):
        if article_id.get("IdType") == "doi":
            doi = article_id.text
            break

    return {
        "pubmed_id": pubmed_id,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "authors": authors,
        "pub_date": pub_date,
        "epub_date": epub_date,
        "doi": doi,
        "pub_types": pub_types,
    }


def retrieve_range(start_date, end_date, output_path=None, journals=None):
    """
    Fetch all candidate articles across all configured journals for a given date range.

    Args:
        start_date: Start of date range (date object)
        end_date: End of date range (date object)
        output_path: Optional path to save results as JSON. If None, results
                     are returned but not saved.
        journals: Optional list of journal dicts to search. Defaults to JOURNALS.

    Returns:
        List of article dicts.
    """
    if journals is None:
        journals = JOURNALS

    all_articles = []
    seen_pmids = set()

    for journal in journals:
        query = build_query(
            journal=journal["journal"],
            neuro_keyword=journal["neuro_keyword"],
            start_date=start_date,
            end_date=end_date,
        )

        print(f"Searching: {journal['journal']}...")
        pmids = search_pubmed(query)

        if not pmids:
            print(f"  No results.")
            continue

        articles = fetch_articles(pmids)

        # Keep only Journal Articles and Reviews — drop comments, news, editorials, etc.
        ALLOWED_TYPES = {"Journal Article", "Review"}
        articles = [
            a for a in articles
            if any(t in ALLOWED_TYPES for t in a["pub_types"])
        ]

        # Deduplicate across journals by PMID
        new_articles = []
        for article in articles:
            if article["pubmed_id"] not in seen_pmids:
                seen_pmids.add(article["pubmed_id"])
                new_articles.append(article)

        print(f"  Found {len(new_articles)} articles.")
        all_articles.extend(new_articles)

    print(f"\nTotal articles retrieved: {len(all_articles)}")

    conn = db.get_connection()
    inserted = db.insert_articles(conn, all_articles)
    conn.close()
    print(f"New articles added to database: {inserted} ({len(all_articles) - inserted} already seen)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(all_articles, f, indent=2)
        print(f"Saved to {output_path}")

    return all_articles


def retrieve_month(year, month, output_path=None, journals=None):
    """
    Convenience wrapper around retrieve_range for a full calendar month.

    Args:
        year: Publication year (e.g. 2026)
        month: Publication month as integer (e.g. 2 for February)
        output_path: Optional path to save results as JSON.
        journals: Optional list of journal dicts to search. Defaults to JOURNALS.

    Returns:
        List of article dicts.
    """
    start_date = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)

    return retrieve_range(start_date, end_date, output_path=output_path, journals=journals)
