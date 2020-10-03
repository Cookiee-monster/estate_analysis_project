--------------------------------------------------------
--  File created - piπtek-paüdziernika-02-2020   
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table OTODOM_OFFERS
--------------------------------------------------------

  CREATE TABLE "GRZKUP_P"."OTODOM_OFFERS" 
   (	"ID" NUMBER, 
	"URL" VARCHAR2(250 BYTE), 
	"SCRAPED" NUMBER(38,0), 
	"SCRAPING_DATE" TIMESTAMP (6)
   ) PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT)
  TABLESPACE "GRZKUP_P" ;
--------------------------------------------------------
--  DDL for Index OTODOM_OFFERS_INDEX1
--------------------------------------------------------

  CREATE UNIQUE INDEX "GRZKUP_P"."OTODOM_OFFERS_INDEX1" ON "GRZKUP_P"."OTODOM_OFFERS" ("ID") 
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT)
  TABLESPACE "GRZKUP_P" ;
--------------------------------------------------------
--  DDL for Trigger OTODOM_OFFERS_TRG
--------------------------------------------------------

  CREATE OR REPLACE TRIGGER "GRZKUP_P"."OTODOM_OFFERS_TRG" 
BEFORE INSERT ON OTODOM_OFFERS 
FOR EACH ROW 
BEGIN
  <<COLUMN_SEQUENCES>>
  BEGIN
    NULL;
  END COLUMN_SEQUENCES;
END;
/
ALTER TRIGGER "GRZKUP_P"."OTODOM_OFFERS_TRG" ENABLE;
--------------------------------------------------------
--  DDL for Trigger OTODOM_OFFERS_TRG_1
--------------------------------------------------------

  CREATE OR REPLACE TRIGGER "GRZKUP_P"."OTODOM_OFFERS_TRG_1" 
BEFORE INSERT ON OTODOM_OFFERS 
FOR EACH ROW
BEGIN
  if :NEW.id is null then 
    select OTODOM_OFFERS_SEQ.nextval into :NEW.id from dual; 
  end if; 
END;
/
ALTER TRIGGER "GRZKUP_P"."OTODOM_OFFERS_TRG_1" ENABLE;
--------------------------------------------------------
--  Constraints for Table OTODOM_OFFERS
--------------------------------------------------------

  ALTER TABLE "GRZKUP_P"."OTODOM_OFFERS" ADD CONSTRAINT "OTODOM_OFFERS_PK" PRIMARY KEY ("ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS 
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT)
  TABLESPACE "GRZKUP_P"  ENABLE;
 
  ALTER TABLE "GRZKUP_P"."OTODOM_OFFERS" MODIFY ("ID" NOT NULL ENABLE);
--------------------------------------------------------
--  Ref Constraints for Table OTODOM_OFFERS
--------------------------------------------------------

  ALTER TABLE "GRZKUP_P"."OTODOM_OFFERS" ADD CONSTRAINT "OTODOM_OFFERS_FK" FOREIGN KEY ("SCRAPED")
	  REFERENCES "GRZKUP_P"."OTODOM_STATUS" ("ID") ENABLE;
