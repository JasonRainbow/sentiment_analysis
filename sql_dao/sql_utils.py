import pymysql
from pymysql.err import ProgrammingError
from sql_dao import host, port, username, password, database, charset


def get_conn():
    return pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database,
        charset=charset
    )


insert_polarity_sql = """
    insert into polarity_analy(workId, country, platform, time, positive, negative, neutrality) 
    values({}, "{}", "{}", "{}", {}, {}, {});
"""

insert_sentiment_sql = """
    insert into sentiment_analy(workId, country, platform, time, happy, amazed, neutrality,
    sad, angry, fear) 
    values({}, "{}", "{}", "{}", {}, {}, {}, {}, {}, {});
"""

delete_sentiment_sql = """
    delete from sentiment_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

query_sentiment_sql = """
    select * from sentiment_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

delete_polarity_sql = """
    delete from polarity_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

query_polarity_sql = """
    select * from polarity_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

# cursor = conn.cursor()
query_sql = """
    select * from user;
"""
# df = pd.read_sql(query_sql, con=conn)
# print(df)


def insert_polarity(workId, country, platform, post_time, positive, negative, neutrality, conn):
    cursor = conn.cursor()
    if positive is None:
        positive = 0
    if negative is None:
        negative = 0
    if neutrality is None:
        neutrality = 0
    try:
        cursor.execute(insert_polarity_sql.format(workId, country, platform, post_time, positive, negative, neutrality))
        conn.commit()
        return True
    except ProgrammingError:
        print("sql语法错误")
        return False
    finally:
        cursor.close()


def insert_sentiment(workId, country, platform, post_time, happy, amazed, neutrality, sad, angry, fear, conn):
    cursor = conn.cursor()
    if happy is None:
        happy = 0
    if amazed is None:
        amazed = 0
    if neutrality is None:
        neutrality = 0
    if sad is None:
        sad = 0
    if angry is None:
        angry = 0
    if fear is None:
        fear = 0
    try:
        cursor.execute(insert_sentiment_sql
                       .format(workId, country, platform, post_time, happy, amazed, neutrality, sad, angry, fear))
        conn.commit()
        return True
    except ProgrammingError:
        print("sql语法错误")
        return False
    finally:
        cursor.close()


if __name__ == "__main__":
    pass
