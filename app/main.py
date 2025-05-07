import os
import json
import mysql.connector  # MySQL driver
from db import mysql_conn, mongo_conn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from bson import ObjectId
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import logging
from typing import Dict
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client: OpenAI | None = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=api_key.strip())
    client.models.list()
    OPENAI_CONFIGURED = True
    logger.info("OpenAI client ready")
except Exception as e:
    OPENAI_CONFIGURED = False
    logger.error(f"OpenAI init failed: {e}")

mongo_db = None
try:
    mongo_db = mongo_conn()
    if mongo_db is not None:
        logger.info(f"MongoDB connected to {mongo_db.name}")
    else:
        logger.warning("MongoDB connection attempt returned None")
except Exception as e:
    logger.error(f"MongoDB init failed: {e}")


class StrJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, default=str).encode("utf-8")


app = FastAPI(
    title="ChatDB API",
    default_response_class=StrJSONResponse,
)


class ChatRequest(BaseModel):
    query: str
    db_type: str


def _dumps(obj):
    return json.dumps(obj)


def mongo_cmd_to_string(cmd: Dict) -> str:
    op_raw = cmd.get("operation", "")
    op = op_raw.replace("_", "").lower()
    coll = cmd.get("collection", "")

    if op == "aggregate":
        pipeline = cmd.get("pipeline", [])
        return f"{coll}.aggregate({_dumps(pipeline)})"

    if op in ("find", "findone"):
        filt = _dumps(cmd.get("filter", {}))
        proj = cmd.get("projection")
        proj_part = f", {_dumps(proj)}" if proj else ""

        base = f"{coll}.{op}({filt}{proj_part})"

        sort = cmd.get("sort")
        if sort:
            sort_part = _dumps(sort)
            base += f".sort({sort_part})"

        for kw in ("skip", "limit"):
            if isinstance(cmd.get(kw), int):
                base += f".{kw}({cmd[kw]})"

        return base

    if op in ("count", "countdocuments"):
        filt = _dumps(cmd.get("filter", {}))
        return f"{coll}.countDocuments({filt})"

    if op == "insertone":
        doc = _dumps(cmd.get("document", {}))
        return f"{coll}.insertOne({doc})"

    if op == "insertmany":
        docs = _dumps(cmd.get("documents", []))
        return f"{coll}.insertMany({docs})"

    if op in ("updateone", "updatemany"):
        filt = _dumps(cmd.get("filter", {}))
        upd = _dumps(cmd.get("update", {}))
        opts = cmd.get("options") or {}
        if opts:
            return f"{coll}.{op_raw}({filt}, {upd}, {_dumps(opts)})"
        return f"{coll}.{op_raw}({filt}, {upd})"

    if op in ("deleteone", "deletemany"):
        filt = _dumps(cmd.get("filter", {}))
        return f"{coll}.{op_raw}({filt})"

    if op in ("listcollections", "showcollections"):
        return "db.getCollectionNames()"

    return _dumps(cmd)


def get_mysql_schema() -> str:
    """Return prettified schema description from INFORMATION_SCHEMA."""
    operation_context = "connect"
    conn = None
    schema_info: dict[str, list[str]] = {}
    try:
        conn = mysql_conn()
        cur = conn.cursor(dictionary=True)

        operation_context = "fetch database()"
        cur.execute("SELECT DATABASE()")
        db_name_row = cur.fetchone()
        if not db_name_row:
            raise RuntimeError("Cannot determine current database")
        db_name = db_name_row["DATABASE()"]

        operation_context = "fetch tables"
        cur.execute(
            """
            SELECT TABLE_NAME AS table_name 
            FROM information_schema.tables
            WHERE table_schema = %s
            """,
            (db_name,),
        )
        tables = cur.fetchall()
        for tbl in tables:
            tbl_name = tbl["table_name"]
            cur.execute(
                """
                SELECT COLUMN_NAME   AS column_name,
                    DATA_TYPE     AS data_type,
                    COLUMN_KEY    AS column_key,
                    IS_NULLABLE   AS is_nullable,
                    COLUMN_DEFAULT AS column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (db_name, tbl_name),
            )

            cols = cur.fetchall()
            col_descs: list[str] = []
            for col in cols:
                desc = f"{col['column_name']} ({col['data_type']}";
                if col["column_key"] == "PRI":
                    desc += ", PK"
                if col["is_nullable"] == "NO":
                    desc += ", NOT NULL"
                if col["column_default"] is not None:
                    desc += f", DEFAULT '{col['column_default']}'"
                desc += ")"
                col_descs.append(desc)
            schema_info[tbl_name] = col_descs
        formatted = [f"Table: {t}\n  Columns: {'; '.join(c)}" for t, c in schema_info.items()]
        return "MySQL Schema:\n" + "\n\n".join(formatted)
    except Exception as e:
        raise RuntimeError(f"{operation_context} failed: {e}") from e
    finally:
        if conn and conn.is_connected():
            conn.close()


def get_mongodb_schema() -> str:
    if mongo_db is None:
        logger.error("Attempted to get MongoDB schema but connection is not available.")
        raise RuntimeError("MongoDB connection not available")
    try:
        info = [f"MongoDB Collections (DB: {mongo_db.name}):"]
        collection_names = mongo_db.list_collection_names()
        logger.info(f"Found MongoDB collections: {collection_names}")
        for coll_name in collection_names:
            sample_doc = "N/A"
            try:
                sample = mongo_db[coll_name].find_one() or {}
                sample.pop("_id", None)
                sample_str = json.dumps(sample, default=str)
                if len(sample_str) > 150:
                    sample_doc = sample_str[:150] + "...}"
                else:
                    sample_doc = sample_str
            except Exception as sample_err:
                logger.warning(f"Could not get sample for collection {coll_name}: {sample_err}")

            info.append(f" - Collection: {coll_name}, Sample fields: {sample_doc}")
        return "\n".join(info)
    except Exception as e:
        logger.error(f"Error listing MongoDB collections or getting samples: {e}")
        raise RuntimeError(f"Failed to retrieve MongoDB schema: {e}") from e


def construct_mysql_prompt(user_query: str, schema: str) -> str:
    return f"""
You are an expert assistant that translates natural language into MySQL.
Schema:
{schema}

User request: "{user_query}"

Return ONLY the SQL statement as a single line.
"""


def construct_mongodb_prompt(user_query: str, schema: str) -> str:
    return f"""
You are an expert assistant that translates natural language into MongoDB PyMongo commands in JSON.
Available Collections and sample docs:
{schema}

User request: \"{user_query}\"

Return only valid JSON object as a single line with keys: operation, collection, and other fields as needed
(filter, projection, update, pipeline, etc.).
Represent **"sort" as an object**, e.g. "sort": {{"name": 1}}. No markdown.
""".strip()


async def generate_query(prompt: str) -> str:
    if not OPENAI_CONFIGURED or client is None:
        raise RuntimeError("OpenAI not configured")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    sql = resp.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = "\n".join(sql.splitlines()[1:-1]).strip()
    return sql.rstrip(";")


async def generate_mongodb_command(prompt: str) -> dict:
    if not OPENAI_CONFIGURED or not client:
        logger.error("OpenAI client not configured, cannot generate MongoDB command.")
        raise RuntimeError("OpenAI not configured")
    try:
        logger.info("Sending prompt to LLM for MongoDB command generation...")

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,

        )
        text = resp.choices[0].message.content.strip()
        logger.info("Received response from LLM for MongoDB command.")
        if text.startswith("```json"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        elif text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()

        logger.info("Attempting to parse LLM response as JSON...")
        parsed_json = json.loads(text)
        logger.info("Successfully parsed LLM response for MongoDB command.")
        return parsed_json

    except json.JSONDecodeError as e:
        error_message = f"Invalid JSON received from LLM: {e}"
        logger.error(error_message)
        logger.error(f"Problematic text from LLM (length {len(text)}):\n---\n{text}\n---")
        raise ValueError(f"{error_message}. See server logs for the full invalid text received.")

    except Exception as e:
        logger.error(f"Error during MongoDB command generation or LLM call: {e}", exc_info=True)  # Log stack trace
        raise RuntimeError(f"Failed to get command from LLM: {e}")


def execute_mysql(sql: str):
    conn = mysql_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)

    if cur.with_rows:
        res = cur.fetchall()
    else:
        conn.commit()
        res = {"rows_affected": cur.rowcount}

    cur.close()
    conn.close()
    return res


def _serialize_doc(doc: dict) -> dict:
    if doc is None:
        return None
    if "_id" in doc and isinstance(doc["_id"], ObjectId):
        doc["_id"] = str(doc["_id"])
    return doc


from bson import ObjectId


def execute_mongodb_command(cmd: dict):
    if mongo_db is None:
        logger.error("MongoDB command execution failed: No active connection.")
        raise RuntimeError("MongoDB is not connected.")

    if not isinstance(cmd, dict):
        logger.error(f"Invalid command format: Expected dict, got {type(cmd)}")
        raise ValueError(f"Invalid command format: Expected dict, got {type(cmd)}. Command: {cmd}")

    op = cmd.get("operation", "").lower()
    norm = op.replace("_", "").replace(" ", "")

    if norm in ("listcollections", "showcollections"):
        logger.info("Listing MongoDB collections")
        return mongo_db.list_collection_names()

    coll_name = cmd.get("collection")
    if not coll_name:
        logger.error(f"Command must include 'collection'. Received: {cmd}")
        raise ValueError(f"Command must include 'collection'. Received: {cmd}")
    coll = mongo_db[coll_name]

    try:
        # findOne
        if norm == "findone":
            doc = coll.find_one(cmd.get("filter", {}), cmd.get("projection"))
            return _serialize_doc(doc)

        # findMany
        if norm == "find":
            cursor = coll.find(cmd.get("filter", {}), cmd.get("projection"))

            sort_arg = cmd.get("sort")
            if sort_arg:
                if isinstance(sort_arg, dict):
                    cursor = cursor.sort(list(sort_arg.items()))
                elif isinstance(sort_arg, list):
                    cursor = cursor.sort(sort_arg)
                else:
                    raise ValueError('"sort" must be dict or list of pairs')

            if isinstance(cmd.get("skip"), int):
                cursor = cursor.skip(cmd["skip"])
            if isinstance(cmd.get("limit"), int):
                cursor = cursor.limit(cmd["limit"])

            return [_serialize_doc(d) for d in cursor]

        # aggregate
        if norm == "aggregate":
            pipeline = cmd.get("pipeline", [])
            if not isinstance(pipeline, list):
                raise ValueError("'pipeline' must be a list for aggregate")
            return [_serialize_doc(d) for d in coll.aggregate(pipeline)]

        # countDocuments
        if norm in ("count", "countdocuments"):
            count = coll.count_documents(cmd.get("filter", {}))
            return {"count": count}

        # insertOne / insertMany
        if norm in ("insertone",):
            res = coll.insert_one(cmd.get("document", {}))
            return {"inserted_id": str(res.inserted_id)}
        if norm in ("insertmany",):
            res = coll.insert_many(cmd.get("documents", []))
            return {"inserted_ids": [str(_id) for _id in res.inserted_ids]}

        # updateOne / updateMany
        if norm in ("updateone",):
            res = coll.update_one(cmd.get("filter", {}), cmd.get("update", {}))
            return {"matched_count": res.matched_count, "modified_count": res.modified_count}
        if norm in ("updatemany",):
            res = coll.update_many(cmd.get("filter", {}), cmd.get("update", {}))
            return {"matched_count": res.matched_count, "modified_count": res.modified_count}

        # deleteOne / deleteMany
        if norm in ("deleteone",):
            res = coll.delete_one(cmd.get("filter", {}))
            return {"deleted_count": res.deleted_count}
        if norm in ("deletemany",):
            res = coll.delete_many(cmd.get("filter", {}))
            return {"deleted_count": res.deleted_count}

        logger.error(f"Unsupported MongoDB operation: {op}")
        raise ValueError(f"Unsupported MongoDB operation: {op}")

    except Exception as e:
        logger.error(f"Error executing MongoDB command {cmd}: {e}", exc_info=True)
        raise RuntimeError(f"MongoDB execution failed: {e}") from e


@app.post("/chat")
async def chat(req: ChatRequest):
    dbt = req.db_type.lower()
    logger.info(f"Received chat request for db_type: {dbt}, query: '{req.query}'")  # Log request

    if dbt == "mysql":
        try:
            logger.info("Getting MySQL schema...")
            schema = get_mysql_schema()
            logger.info("Constructing MySQL prompt...")
            prompt = construct_mysql_prompt(req.query, schema)
            logger.info("Generating MySQL query via LLM...")
            sql = await generate_query(prompt)
            logger.info(f"Generated SQL: {sql}")
            logger.info("Executing MySQL query...")
            res = await asyncio.to_thread(execute_mysql, sql)
            logger.info("MySQL execution successful.")
            return {"generated_sql": sql, "results": res}
        except Exception as e:
            logger.error(f"Error processing MySQL request: {e}", exc_info=True)
            detail = f"Failed to process MySQL request: {e}"
            if isinstance(e, mysql.connector.Error):
                detail = f"MySQL Error: {e}"
            elif isinstance(e, RuntimeError) and "OpenAI" in str(e):
                detail = f"LLM Error: {e}"

            raise HTTPException(status_code=500, detail=detail)

    elif dbt == "mongodb":
        if mongo_db is None:
            logger.error("MongoDB request failed: Connection not available.")
            raise HTTPException(status_code=503, detail="MongoDB connection is not available")
        try:
            logger.info("Getting MongoDB schema...")
            schema = get_mongodb_schema()
            logger.info("Constructing MongoDB prompt...")
            prompt = construct_mongodb_prompt(req.query, schema)
            logger.info("Generating MongoDB command via LLM...")
            cmd = await generate_mongodb_command(prompt)
            logger.info(f"Generated MongoDB command: {cmd}")
            cmd_str = mongo_cmd_to_string(cmd).replace('"', "'")
            logger.info("Executing MongoDB command...")
            res = execute_mongodb_command(cmd)
            logger.info("MongoDB execution successful.")
            return StrJSONResponse(content={"generated_command": cmd_str, "results": res})
        except ValueError as e:
            logger.error(f"Error processing MongoDB request (ValueError): {e}", exc_info=False)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing MongoDB request: {e}", exc_info=True)
            detail = f"Failed to process MongoDB request: {e}"
            if isinstance(e, RuntimeError) and ("OpenAI" in str(e) or "LLM" in str(e)):
                detail = f"LLM Error: {e}"
            elif isinstance(e, RuntimeError) and "MongoDB execution" in str(e):
                detail = f"MongoDB Execution Error: {e}"
            raise HTTPException(status_code=500, detail=detail)
    else:
        logger.warning(f"Invalid db_type received: {req.db_type}")
        raise HTTPException(status_code=400, detail="db_type must be 'mysql' or 'mongodb'")


@app.get("/")
def root():
    openai_status = "OK" if OPENAI_CONFIGURED else "Not Configured"
    mysql_status = "Unknown"
    try:
        conn = mysql_conn()
        conn.ping(reconnect=False)
        mysql_status = "OK"
        conn.close()
    except Exception as e:
        logger.warning(f"MySQL status check failed: {e}")
        mysql_status = "Connection Failed"

    mongo_status = "Unknown"
    if mongo_db is not None:
        try:
            mongo_db.admin.command('ping')
            mongo_status = "OK"
        except Exception as e:
            logger.warning(f"MongoDB status check failed: {e}")
            mongo_status = "Connection Failed"
    else:
        mongo_status = "Not Connected"

    return {
        "message": "ChatDB API is running",
        "openai_status": openai_status,
        "mysql_status": mysql_status,
        "mongodb_status": mongo_status
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
